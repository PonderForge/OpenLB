use fast_image_resize::Resizer;
use image::imageops::overlay;
use image::ImageBuffer;
use image::Rgb;
use ndarray::ArrayBase;
use ndarray::Dim;
use ndarray::OwnedRepr;
use ndarray::{s, Array, Axis, IxDyn};
use crate::img_filter::box_mbr::Mbr;
use ort::{Session, inputs};
use image::DynamicImage;
use nshare::AsNdarray3;
use fast_image_resize::ResizeOptions;

pub fn detect_humans(detector: &Session, input_img: &DynamicImage, resize_options: &ResizeOptions) -> Vec<(f32, f32, f32, f32, f32)> {
    //Convert Image to a Tensor
    let input_tensor = ort::Tensor::from_array(obj_preprocess(&input_img, &resize_options)).unwrap();
    //Run the Human Detector (YOLOv11) on the Image Tensor
    let output_tensor = detector.run(inputs!["images" => input_tensor].unwrap()).unwrap();
    let outputs = output_tensor["output0"].try_extract_tensor::<f32>().unwrap().into_owned();
    return obj_postprocess(vec![outputs], &input_img, 0.45);
}

pub fn detect_humans_warmup (detector: &Session) {
    let input_tensor = ort::Tensor::from_array(([1usize,3,640,640], vec![0.0f32; 1228800])).unwrap();
    let _ = detector.run(inputs!["images" => input_tensor].unwrap()).unwrap();
}


fn obj_preprocess(input: &DynamicImage, resize_options: &ResizeOptions) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let h1 = 640f32 * (input.height() as f32/input.width() as f32);
    let w1 = 640f32 * (input.width() as f32/input.height() as f32);
    let mut dst_image = DynamicImage::new_rgb8(1, 1);
    let mut husk = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(640, 640, Rgb([143u8, 143u8, 143u8])));
    let mut resizer = Resizer::new();
    let mut x1 = 0f32;
    let mut y1 = 0f32;
    if h1 <= 640f32 {
        dst_image = DynamicImage::new(640, h1 as u32, input.color());
        y1 = (640f32 - h1) / 2f32;
    } else {
        dst_image = DynamicImage::new(w1 as u32, 640, input.color());
        x1 = (640f32 - w1) / 2f32;
    }
    resizer.resize(input, &mut dst_image, Some(resize_options)).unwrap();
    overlay(&mut husk, &dst_image, x1 as i64, y1 as i64);
    let array = husk.to_rgb32f().as_ndarray3().to_owned();
    array.insert_axis(Axis(0))
}

fn obj_postprocess( xs: Vec<Array<f32, IxDyn>>, xs0: &DynamicImage, conf: f32 ) -> Vec<(f32, f32, f32, f32, f32)> {
    const CXYWH_OFFSET: usize = 4; // cxcywh
    let preds = &xs[0];
    let anchor = preds.axis_iter(Axis(0)).enumerate().next().unwrap().1;
    // [bs, 4 + nc + nm, anchors]
    // input image
    let width_original = xs0.width() as f32;
    let height_original = xs0.height() as f32;
    let ratio = (640 as f32 / width_original)
        .min(640 as f32 / height_original);

    // save each result
    let mut data: Vec<(f32, f32, f32, f32, f32)> = Vec::new();
    for pred in anchor.axis_iter(Axis(1)) {
        // split preds for different tasks
        let bbox = pred.slice(s![0..CXYWH_OFFSET]);
        let clss = pred.slice(s![CXYWH_OFFSET..CXYWH_OFFSET + 1 as usize]);
        //let rad = pred.slice(s![CXYWH_OFFSET + 1..CXYWH_OFFSET + 2 as usize]);
        
        // confidence and id
        let (_id, &confidence) = clss
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
            x.max(0f32).min(width_original),
            y.max(0f32).min(height_original),
            w,
            h,
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
fn nms(xs: &mut Vec<(f32, f32, f32, f32, f32)>, iou_threshold: f32 ) {
    xs.sort_by(|b1, b2| b2.4.partial_cmp(&b1.4).unwrap());

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