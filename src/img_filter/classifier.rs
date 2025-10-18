use std::time::Instant;

use fast_image_resize::ResizeOptions;
use image::DynamicImage;
use ndarray::{concatenate, s, ArrayBase, ArrayD, Axis, Dim, OwnedRepr};
use nshare::AsNdarray3;
use ort::{inputs, Session, SessionOutputs};
use fast_image_resize::Resizer;

//Runs NSFW Classification on a image or part of image
pub fn classify_images(model: &Session, input_images: &Vec<DynamicImage>, resize_options: &ResizeOptions)  -> ArrayD<f32>{
    let mut img_iter = input_images.iter();
    let mut array = image_process(img_iter.next().unwrap(), resize_options);
    for img in img_iter {
        array = concatenate(Axis(0), &[array.view(), image_process(img, resize_options).view()]).unwrap();
    }
    let input_tensor = ort::Tensor::from_array(array).unwrap();
    let output_tensor: SessionOutputs = model.run(inputs!["input" => input_tensor].unwrap()).unwrap();
    return output_tensor["output"].try_extract_tensor::<f32>().unwrap().to_owned();
}

pub fn classify_img_warmup (model: &Session) {
    let input_tensor = ort::Tensor::from_array(([1usize,3,384,384], vec![0.0f32; 442368])).unwrap();
    let _ = model.run(inputs!["input" => input_tensor].unwrap()).unwrap();
}

fn image_process (input: &DynamicImage, resize_options: &ResizeOptions) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let mut dst_image = DynamicImage::new(384, 384, input.color());
    let mut resizer = Resizer::new();
    resizer.resize(input, &mut dst_image, Some(resize_options)).unwrap();
    let image = dst_image.into_rgb32f();
    image.as_ndarray3().to_owned().insert_axis(Axis(0))
}