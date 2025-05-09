use std::f32::consts::E;

use opencv::core::*;
use opencv::dnn::*;
use ort::{inputs, Session, SessionOutputs};

//Runs NSFW Classification on a image or part of image
pub fn classify_images (model: &Session, image: &Vector<Mat>) -> Vec<Vec<f32>> {
    //Reformat Image to Classifer Model Input
    let resized_img: Mat = blob_from_images_with_params(&image, Image2BlobParams::new(Scalar::new(0.0171247538317,0.0175070028011,0.0174291938998,0.0), Size::new(384, 384), Scalar::new(123.675, 116.28, 103.53, 0.0), true, CV_32F, DataLayout::DNN_LAYOUT_NCHW, ImagePaddingMode::DNN_PMODE_NULL).unwrap()).unwrap();
    let input_tensor = ort::Tensor::from_array(([image.len() as usize,3,384,384], resized_img.data_typed::<f32>().unwrap())).unwrap();
    //Perform Inference
    let output_tensor: SessionOutputs = model.run(inputs!["input" => input_tensor].unwrap()).unwrap();
    let outputs = output_tensor["output"].try_extract_tensor::<f32>().unwrap().into_owned();
    return outputs.into_raw_vec_and_offset().0.chunks(5).collect::<Vec<&[f32]>>().iter().map(|&e| softmax(e.to_vec())).collect::<Vec<Vec<f32>>>();
}

pub fn classify_warmup (model: &Session) {
    let input_tensor = ort::Tensor::from_array(([1usize,3,384,384], blob_from_image(&Mat::zeros(384, 384, CV_8UC3).unwrap().to_mat().unwrap(), 1f64/255f64, Size::new(384, 384), Scalar::new(0.0,0.0,0.0,0.0), false, false, CV_32F).unwrap().data_typed::<f32>().unwrap())).unwrap();
    let _ = model.run(inputs!["input" => input_tensor].unwrap()).unwrap();
}

fn softmax(array: Vec<f32>) -> Vec<f32> {
    let mut softmax_array = array;

    for value in &mut softmax_array {
        *value = E.powf(*value);
    }

    let sum: f32 = softmax_array.iter().sum();

    for value in &mut softmax_array {
        *value /= sum;
    }

    softmax_array
}