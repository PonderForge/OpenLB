use std::time::Instant;

use opencv::core::*;
use opencv::imgproc::*;
use opencv::dnn::*;
use opencv::imgcodecs::imwrite;
use ort::{inputs, Session, SessionOutputs};

pub enum MaskType {
    Torso,
    Thighs,
}

pub fn segment_image(segmenter: &Session, input_img: &Vector<Mat>) -> Vec<[Result<RotatedRect, ()>; 2]> {
    //Convert Image to a Float Array
    let img_array = blob_from_images_with_params(&input_img, Image2BlobParams::new(Scalar::new(0.00390625,0.00390625,0.00390625,0.0), Size::new(512, 512), Scalar::new(128.0, 128.0, 128.0, 0.0), true, CV_32F, DataLayout::DNN_LAYOUT_NCHW, ImagePaddingMode::DNN_PMODE_LETTERBOX).unwrap()).unwrap();
    let input_tensor = ort::Tensor::from_array(([input_img.len(),3usize,512,512], img_array.data_typed::<f32>().unwrap())).unwrap();
    
    //Run Segmentor
    let inference_result: SessionOutputs = segmenter.run(inputs!["x" => input_tensor].unwrap()).unwrap();
    let result = inference_result["save_infer_model/scale_0.tmp_0"].try_extract_tensor::<f32>().unwrap().into_owned().squeeze();

    let now = Instant::now();
    //Convert Output Tensor to Image 
    let output_vec = result.into_raw_vec_and_offset().0;
    let out_chunks: Vec<&[f32]> = output_vec.chunks(786432).collect();

    let mut output: Vec<[Result<RotatedRect, ()>; 2]> = Vec::new();
    for i in 0..input_img.len() {
        let image = input_img.get(i).unwrap();
        let width = image.cols();
        let height = image.rows();
        output.push([extract_rect_and_image(&image, out_chunks[i], MaskType::Torso, 0.5, width, height, i), extract_rect_and_image(&image, out_chunks[i], MaskType::Thighs, 0.5, width, height, i)]);
    }
    println!("Segment Post-Process Time: {:?}", now.elapsed());
    output
}

pub fn segment_warmup (segmenter: &Session) {
    let input_tensor = ort::Tensor::from_array(([1usize,3,512,512], blob_from_image(&Mat::zeros(512, 512, CV_8UC3).unwrap().to_mat().unwrap(), 1f64/255f64, Size::new(512, 512), Scalar::new(0.0,0.0,0.0,0.0), false, false, CV_32F).unwrap().data_typed::<f32>().unwrap())).unwrap();
    let _ = segmenter.run(inputs!["x" => input_tensor].unwrap()).unwrap();
}

fn extract_rect_and_image (input_img: &Mat, raw_mask: &[f32], masktype: MaskType, thres: f64, width: i32, height: i32, num: usize) -> Result<RotatedRect, ()> {
    let mask_num = match masktype {
        MaskType::Torso => 2,
        MaskType::Thighs => 1,
    };
    let out_chunks: Vec<&[f32]> = raw_mask.chunks(262144).collect();
    let map = Mat::new_rows_cols_with_data( 512, 512, &out_chunks[mask_num]).unwrap();

    let ratio = 512.0 / width.max(height) as f32;
    let cropped = map.roi(Rect::new((512-(width as f32*ratio) as i32)/2, (512-(height as f32*ratio) as i32)/2, (width as f32*ratio) as i32, (height as f32*ratio) as i32)).unwrap();

    //convert from f32 to u8
    let mut image = Mat::default();
    multiply(&cropped, &Scalar::new(256.0, 256.0, 256.0, 0.0), &mut image, 1.0, -1);

    //Create Binary Mask
    let mut mask = Mat::default();
    threshold(&image, &mut mask, thres*256.0, 255.0, THRESH_BINARY);
    let mut resized = Mat::default();
    let do_resize = width.max(height) < 512;
    if do_resize {
        resize(&mask, &mut resized, Size::new(width, height), 0.0, 0.0, INTER_NEAREST);
    }   
    //Create Bounding Box of Partition
    let mut points = Mat::default();
    find_non_zero(if do_resize {&resized} else {&mask}, &mut points);
    let rect_pre = min_area_rect(&points);
    if rect_pre.is_err()  {
        return Err(());
    }
    let mut rect = rect_pre.unwrap();
    if !do_resize {
        let ratio = width.max(height) as f32 / 512.0;
        rect.center.x = rect.center.x * ratio;
        rect.center.y = rect.center.y * ratio;
        rect.size.width = rect.size.width * ratio;
        rect.size.height = rect.size.height * ratio;
    }

    //Debug Mask Bounding
    imwrite(&format!("mask_{}_{}.jpg", num, mask_num), &image, &opencv::core::Vector::new());
    Ok(rect)

}