use opencv::core::*;
use opencv::imgproc::*;
use opencv::dnn::*;
use ndarray::{s, Array, Axis, IxDyn};
use mbr::Mbr;
use ort::{Session, inputs};

pub mod mbr;

pub fn detect_image(detector: &Session, input_img: &Mat) -> Vec<(f32, f32, f32, f32, usize, f32)> {
    //Convert Image to a Tensor
    let input_tensor = ort::Tensor::from_array(([1usize,3,640,640], obj_preprocess(&input_img).data_typed::<f32>().unwrap())).unwrap();
    //Run the Human Detector (YOLOv11) on the Image Tensor
    let output_tensor = detector.run(inputs!["images" => input_tensor].unwrap()).unwrap();
    let outputs = output_tensor["output0"].try_extract_tensor::<f32>().unwrap().into_owned();
    return obj_postprocess(vec![outputs], &input_img, 0.45);
}
pub fn detect_warmup (detector: &Session) {
    let input_tensor = ort::Tensor::from_array(([1usize,3,640,640], blob_from_image(&Mat::zeros(640, 640, CV_8UC3).unwrap().to_mat().unwrap(), 1f64/255f64, Size::new(640, 640), Scalar::new(0.0,0.0,0.0,0.0), false, false, CV_32F).unwrap().data_typed::<f32>().unwrap())).unwrap();
    let _ = detector.run(inputs!["images" => input_tensor].unwrap()).unwrap();
}

//Preprocess YOLO image for inference
pub fn obj_preprocess(input: &Mat) -> Mat {
    let mut output: Mat = Mat::default();

    let h1 = 640f32 * (input.rows() as f32/input.cols() as f32);
    let w1 = 640f32 * (input.cols() as f32/input.rows() as f32);
    if h1 <= 640f32 {
        resize( input, &mut output, opencv::core::Size_::new(640, h1 as i32), 0.0, 0.0, INTER_LINEAR);
    } else {
        resize( input, &mut output, opencv::core::Size_::new(w1 as i32, 640), 0.0, 0.0, INTER_LINEAR);
    }

    let top = (640-output.rows()) / 2;
    let down = (640-output.rows()+1) / 2;
    let left = (640- output.cols()) / 2;
    let right = (640 - output.cols()+1) / 2;
    let mut out: Mat = Mat::default();
    copy_make_border(&output, &mut out, top, down, left, right, BORDER_CONSTANT, opencv::core::Scalar::new(144.0, 144.0, 144.0, 0.0) );
    blob_from_image(&out, 1f64/255f64, Size::new(640, 640), Scalar::new(0.0,0.0,0.0,0.0), true, false, CV_32F).unwrap()
}

//Postprocess YOLO boxes
pub fn obj_postprocess( xs: Vec<Array<f32, IxDyn>>, xs0: &Mat, conf: f32 ) -> Vec<(f32, f32, f32, f32, usize, f32)> {
    const CXYWH_OFFSET: usize = 4; // cxcywh
    let preds = &xs[0];
    let anchor = preds.axis_iter(Axis(0)).enumerate().next().unwrap().1;
    // [bs, 4 + nc + nm, anchors]
    // input image
    let width_original = xs0.cols() as f32;
    let height_original = xs0.rows() as f32;
    let ratio = (640 as f32 / width_original)
        .min(640 as f32 / height_original);

    // save each result
    let mut data: Vec<(f32, f32, f32, f32, usize, f32)> = Vec::new();
    for pred in anchor.axis_iter(Axis(1)) {
        // split preds for different tasks
        let bbox = pred.slice(s![0..CXYWH_OFFSET]);
        let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + 1 as usize]);
        //let rad = pred.slice(s![CXYWH_OFFSET + 1..CXYWH_OFFSET + 2 as usize]);
        
        // confidence and id
        let (id, &confidence) = clss
            .into_iter()
            .enumerate()
            .reduce(|max, x| if x.1 > max.1 { x } else { max })
            .unwrap(); // definitely will not panic!

        // confidence filter
        if confidence < conf {
            continue;
        }
        let square_max = width_original.max(height_original);
        // bbox re-scale
        let cx = bbox[0] / ratio;
        let cy = bbox[1] / ratio;
        let w = bbox[2] / ratio;
        let h = bbox[3] / ratio;
        let x = (cx - w / 2.) - ((square_max-width_original)/2.0);
        let y = (cy - h / 2.) - ((square_max-height_original)/2.0);
        let y_bbox = (
            x.max(0.0f32).min(width_original),
            y.max(0.0f32).min(height_original),
            w,
            h,
            id,
            confidence,
        );

        // data merged
        data.push(y_bbox);
    }

    // nms
    nms(&mut data, 0.40);
    data
}

//Rotated NMS function
fn nms(xs: &mut Vec<(f32, f32, f32, f32, usize, f32)>, iou_threshold: f32 ) {
    xs.sort_by(|b1, b2| b2.5.partial_cmp(&b1.5).unwrap());

    let mut current_index = 0;
    for index in 0..xs.len() {
        let mut drop = false;
        for prev_index in 0..current_index {
            let mbr = Mbr::from_cxcywhr((xs[index].0 + (xs[index].2/2.0)) as f64, (xs[index].1 + (xs[index].3/2.0)) as f64, xs[index].2 as f64, xs[index].3 as f64, 0.0);
            let mbr2 = Mbr::from_cxcywhr((xs[prev_index].0 + (xs[prev_index].2/2.0)) as f64, (xs[prev_index].1 + (xs[prev_index].3/2.0)) as f64, xs[prev_index].2 as f64, xs[prev_index].3 as f64, 0.0);
            let iou = mbr.iou(&mbr2);
            if iou > iou_threshold {
                drop = true;
                break;
            }
        }
        if !drop {
            xs.swap(current_index, index);
            current_index += 1;
        }
    }
    xs.truncate(current_index);
}