mod object;
mod partition;
mod classifier;

use object::{detect_image, detect_warmup};
use opencv::imgcodecs::imwrite;
use opencv::imgcodecs::IMREAD_UNCHANGED;
use ort::ExecutionProviderDispatch;
use partition::{segment_image, segment_warmup};
use classifier::{classify_images, classify_warmup};
use ort::GraphOptimizationLevel;
use ort::Session;
use opencv::imgcodecs::imdecode;
use opencv::core::*;
use opencv::imgproc::*;
use std::time::Instant;
#[cfg(feature = "bincode")]
use bincode::{Encode, Decode};

#[derive(Debug)]
pub struct LBCleaner {
    detector: Session,
    classifier: Session,
    segmenter: Session,
    human_thresholds: LBThresholds,
    overall_thresholds: LBThresholds,
    part_thresholds: LBThresholds,
}

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct LBThresholds {
    pub sexy: f32,
    pub porn: f32,
    pub hentai: f32,
}

impl LBThresholds {
    pub fn new (sexy_threshold: f32, porn_threshold: f32, hentai_threshold: f32) -> LBThresholds {
        LBThresholds {sexy: sexy_threshold, porn: porn_threshold, hentai: hentai_threshold}
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub enum CleanLevel {
    Overall,
    Human,
    Parts,
    OriginalLB
}

pub fn init_defaults() -> LBCleaner {
    //Initialize Onnx Runtime
    let ort_init = ort::init().commit();
    if ort_init.is_err() {
        panic!("ONNX was not correctly initalized!");
    }

    //Load Models
    let detector = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../detect.onnx")).unwrap();
    let classifier = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../classify.onnx")).unwrap();
    let segmenter = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../segment.onnx")).unwrap();
    let thresholds = LBThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
    LBCleaner { detector: detector, classifier: classifier, segmenter: segmenter, human_thresholds: thresholds, overall_thresholds: thresholds, part_thresholds: thresholds }
}

pub fn init(human_thresholds: LBThresholds, overall_thresholds: LBThresholds, part_thresholds: LBThresholds, exec_providers: ExecutionProviderDispatch) -> LBCleaner {
    //Initialize Onnx Runtime
    let ort_init = ort::init()
    .with_execution_providers([exec_providers])
    .commit();
    if ort_init.is_err() {
        panic!("ONNX was not correctly initalized!");
    }

    //Load Models
    let detector = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../detect.onnx")).unwrap();
    let classifier = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../classify.onnx")).unwrap();
    let segmenter = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../segment.onnx")).unwrap();
    LBCleaner { detector: detector, classifier: classifier, segmenter: segmenter, human_thresholds: human_thresholds, overall_thresholds: overall_thresholds, part_thresholds: part_thresholds }
}

impl LBCleaner {
    pub fn warmup(&self, iters: u8) {
        //Warmup Models
        for _ in 0..iters {
            detect_warmup(&self.detector);
            classify_warmup(&self.classifier);
            segment_warmup(&self.segmenter);
        }
    }

    pub fn clean_file_path(&self, input_path: &str, output_path: &str, level: CleanLevel) {
        let input_img = &opencv::imgcodecs::imread(input_path, IMREAD_UNCHANGED).unwrap();
        let out = self.clean_mat(&input_img, level);
        if out.is_some() {
            opencv::imgcodecs::imwrite(output_path, &out.unwrap(), &opencv::core::Vector::new()).unwrap();
        }
    }

    pub fn clean_mat(&self, input_img: &Mat, level: CleanLevel) -> Option<Mat> {
        let mut exit_img = input_img.clone();
        let mut changed = false;
        let mut processed_for_ai = Mat::default();
        cvt_color_def(&input_img, &mut processed_for_ai, COLOR_BGRA2BGR).unwrap();
        let now = Instant::now();
        println!("Start Inference");
        if level == CleanLevel::Overall {
            let metric = classify_images(&self.classifier, &Vector::from_elem(processed_for_ai.clone(), 1));
            println!("Level 0 Time: {:?}", now.elapsed());
            if metric[0][4] > self.human_thresholds.sexy || metric[0][1] > self.human_thresholds.hentai || metric[0][3] > self.human_thresholds.porn {
                println!("Detected NSFW Content");
                exit_img = create_overlay(exit_img.cols(), exit_img.rows(), Scalar::new( metric[0][3] as f64 * 200.0, metric[0][1] as f64 * 200.0, metric[0][4] as f64 * 200.0, 0.0), input_img.channels());
                changed = true;
            }
        } else {
            //Run Human Detector and convert to Vector
            let mut humans: Vec<(f32, f32, f32, f32, usize, f32)> = detect_image(&self.detector, &processed_for_ai);
            //Load Images into Vector
            let mut mats: Vector<Mat> = Vector::new();
            for human in &humans {
                let mut cropped: Mat = Mat::default();
                get_rect_sub_pix_def(&processed_for_ai, Size::new(human.2 as i32, human.3 as i32), Point2f::new(human.0 + (human.2/2.0), human.1 + (human.3/2.0)), &mut cropped).unwrap();
                mats.push(cropped);
            }
            println!("Human Time: {:?}", now.elapsed());
            if !humans.is_empty() {
                if level == CleanLevel::Human || level == CleanLevel::OriginalLB {
                    let now2 = Instant::now();
                    let human_metrics = classify_images(&self.classifier, &mats);
                    println!("Classify Time: {:?}", now2.elapsed());
                    for i in 0..humans.len() {
                        println!("Human Metric: {:?}", human_metrics[i]);
                        //rectangle(&mut exit_img, Rect::from_point_size(Point::new(humans[i].0 as i32, humans[i].1 as i32), Size::new(humans[i].2 as i32 - 5, humans[i].3 as i32 - 5)), Scalar::new(human_metrics[i][1] as f64 * 255.0, 0.0, 0.0, 0.0), -1, LINE_8, 0);
                        if human_metrics[i][4] > self.overall_thresholds.sexy || human_metrics[i][1] > self.overall_thresholds.hentai || human_metrics[i][3] > self.overall_thresholds.porn {
                            println!("Detected NSFW Content");
                            if humans[i].0 + humans[i].2 > exit_img.cols() as f32 {
                                humans[i].2 += exit_img.cols() as f32 - (humans[i].0 + humans[i].2);
                            }
                            if humans[i].1 + humans[i].3 > exit_img.rows() as f32 {
                                humans[i].3 += exit_img.rows() as f32 - (humans[i].1 + humans[i].3);
                            }
                            let overlay = create_overlay(humans[i].2 as i32, humans[i].3 as i32, Scalar::new( human_metrics[i][3] as f64 * 200.0, human_metrics[i][1] as f64 * 200.0, human_metrics[i][4] as f64 * 200.0, 0.0), input_img.channels());
                            overlay.copy_to(&mut Mat::roi_mut(&mut exit_img, Rect_ { x: humans[i].0 as i32, y: humans[i].1 as i32, width: humans[i].2 as i32, height: humans[i].3 as i32}).unwrap()).unwrap();
                            changed = true;
                        }
                    }
                    println!("Level 1 Time: {:?}", now.elapsed());
                } else { 
                    //Segmentation Process
                    let image_part_rects = segment_image(&self.segmenter, &mats);
                    println!("Segment Time: {:?}", now.elapsed());
                    let flat = image_part_rects.iter().flatten().collect::<Vec<&Result<RotatedRect, ()>>>();
                    //Grab images from bounding box
                    let mut images = Vector::new();
                    for i in &flat {
                        if i.is_err() {
                            continue;
                        }
                        let input = i.unwrap();
                        let matrix = get_rotation_matrix_2d(input.center, input.angle as f64, 1.0).unwrap();
                        let mut affine = Mat::default();
                        warp_affine_def(&input_img, &mut affine, &matrix, Size::new(input_img.cols(), input_img.rows())).unwrap();
                        let mut output = Mat::default();
                        get_rect_sub_pix_def(&affine, Size::new(input.size.width as i32, input.size.height as i32), input.center, &mut output).unwrap();
                        images.push(output);
                    }

                    //Classify images in bulk
                    let now2 = Instant::now();
                    let part_metrics = classify_images(&self.classifier, &images);
                    println!("Segment Class Time: {:?}", now2.elapsed());

                    //Recreate Bounding Box on Input Image with Metrics
                    let mut c = 0;
                    for a in 0..image_part_rects.len() {
                        for b in 0..2 {
                            if image_part_rects[a][b].is_err() {
                                continue;
                            }
                            println!("Part Metric: {:?}", part_metrics[c].clone());
                            let mut vertices: [Point_<f32>; 4]= [Default::default(); 4];
                            image_part_rects[a][b].unwrap().points(&mut vertices).unwrap();
                            let mut svertices: [Point_<i32>; 4]= [Default::default(); 4];
                            for i in 0..4 {
                                //line(&mut exit_img, Point::new((humans[a].0 + vertices[i].x) as i32, (humans[a].1 + vertices[i].y) as i32), Point::new((humans[a].0 + vertices[(i+1)%4].x) as i32, (humans[a].1 + vertices[(i+1)%4].y) as i32), Scalar::new(part_metrics[c].clone()[0] as f64 * 255.0, 0.0, 0.0, 0.0), 10, LINE_8, 0);
                                svertices[i] = Point_::<i32>::new((humans[a].0 + vertices[i].x) as i32, (humans[a].1 + vertices[i].y) as i32);          
                            }
                            
                            let part_metric = part_metrics[c].clone();
                            if part_metric[4] > self.part_thresholds.sexy || part_metric[1] > self.part_thresholds.hentai || part_metric[3] > self.part_thresholds.porn {
                                println!("Detected NSFW Content");
                                fill_convex_poly_def(&mut exit_img, &Vector::from_slice(&svertices), Scalar::new( part_metric[3] as f64 * 200.0, part_metric[1] as f64 * 200.0,part_metric[4] as f64 * 200.0, 0.0)).unwrap();
                                changed = true;
                            }
                            c+=1;
                        }
                    }
                    
                    println!("Level 2 Time: {:?}", now.elapsed());
                }
            } else if level == CleanLevel::OriginalLB {
                let overall_output = self.clean_mat(input_img, CleanLevel::Overall);
                if overall_output.is_some() {
                    exit_img = overall_output.unwrap();
                    changed = true;
                }
            }
        }
        if changed {
            return Some(exit_img);
        } else {
            return None;
        }
    }

    #[cfg(feature = "gif")]
    //Automatic GIF Cleaning Level is Overall, since any other level would probably crash your computer or signifgantly slow it down
    pub fn clean_gif<R> (&self, input_gif: R) -> Option<Vec<u8>> where R: std::io::Read {
        use std::ffi::c_void;

        use gif::Frame;

        let mut decoder = gif::DecodeOptions::new();
        decoder.set_color_output(gif::ColorOutput::RGBA);
        let decoder = decoder.read_info(input_gif).unwrap();
        let width = decoder.width() as i32;
        let height = decoder.height() as i32;
        let colormap = decoder.global_palette().unwrap_or(&[]).to_owned();
        let repeat = decoder.repeat();
        let mut src = Mat::zeros(height, width, CV_8UC4).unwrap().to_mat().unwrap();
        let mut cleaned_frames: Vec<Frame> = Vec::new();
        let mut changed = false;
        let mut decoder_iter = decoder.into_iter();
        while let Some(Ok(frame)) = decoder_iter.next() {
            let data = &frame.buffer;
            let delay = frame.delay;
            let image_raw = unsafe {
                Mat::new_rows_cols_with_data_unsafe_def(height, width, CV_8UC4, data.as_ptr() as *mut c_void).unwrap()
            };
            let mut image = Mat::default();
            cvt_color_def(&image_raw, &mut image, COLOR_RGBA2BGRA).unwrap();
            let mut img_layers: Vector<Mat> = Vector::new();
            split(&image, &mut img_layers);
            image.copy_to_masked(&mut src, &img_layers.get(3).unwrap()).unwrap();
            let cleaned = self.clean_mat(&src, CleanLevel::Human);
            if cleaned.is_some() {
                let mut image_raw = Mat::default();
                cvt_color_def(&cleaned.unwrap(), &mut image_raw, COLOR_BGRA2RGB).unwrap();
                let mut frame = gif::Frame::from_rgb_speed(width as u16, height as u16, &mut image_raw.data_bytes().unwrap().to_vec(), 15);
                if delay < 10 {
                    let skip_frames = (10.0/delay as f32) as usize;
                    decoder_iter.nth(skip_frames);
                    frame.delay = delay * (skip_frames +1) as u16;
                } else {
                    frame.delay = delay;
                }
                cleaned_frames.push(frame);
                changed = true;
            } else {
                let mut image_raw = Mat::default();
                cvt_color_def(&src, &mut image_raw, COLOR_BGRA2RGB).unwrap();
                let mut frame2 = gif::Frame::from_rgb_speed(width as u16, height as u16, &mut image_raw.data_bytes().unwrap().to_vec(), 15);
                frame2.delay = delay;
                cleaned_frames.push(frame2);
            }
        }
        if changed {
            let mut out_file: Vec<u8> = Vec::new();
            {
                let mut encoder = gif::Encoder::new(&mut out_file, width as u16, height as u16, &colormap).unwrap();
                encoder.set_repeat(repeat).unwrap();
                for state in &cleaned_frames {
                    encoder.write_frame(&state).unwrap();
                }
            }
            return Some(out_file);
        } else {
            return None;
        }
    }
}

//Create Overlay for NSFW Content
fn create_overlay (width: i32, height: i32, color: Scalar, channels: i32) -> Mat {
    let mut husk = Mat::zeros(64, 64, CV_8UC4).unwrap().to_mat().unwrap();
    husk.set_to_def(&color).unwrap();
    let icon = imdecode(include_bytes!("../icon.png"), IMREAD_UNCHANGED).unwrap();
    let mut icon_layers: Vector<Mat> = Vector::new();
    split(&icon, &mut icon_layers).unwrap();
    icon.copy_to_masked(&mut husk.roi_mut(Rect_ { x: 16, y: 16, width: 32, height: 32 }).unwrap(), &icon_layers.get(3).unwrap()).unwrap();
    let ratio = 64.0 / width.min(height) as f32;
    let mut extended = Mat::default();
    let extension = (((width.max(height)) as f32 * ratio)/2.0)as i32 -32;
    if width > height {
        copy_make_border(&husk, &mut extended, 0, 0, extension, extension, BORDER_CONSTANT, color).unwrap();
    } else {
        copy_make_border(&husk, &mut extended, extension, extension, 0, 0, BORDER_CONSTANT, color).unwrap();
    }
    if ratio != 1.0 {
        let holder = extended.clone();
        resize(&holder, &mut extended, Size::new(width, height), 0.0, 0.0, INTER_NEAREST).unwrap();
    }
    let mut mask = Mat::default();
    if channels == 1 {
        cvt_color_def(&extended, &mut mask, COLOR_BGRA2GRAY).unwrap();
    } else if channels == 3 {
        cvt_color_def(&extended, &mut mask, COLOR_BGRA2BGR).unwrap();
    } else {
        mask = extended.clone();
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs::{imwrite, imread};
    use ort::CPUExecutionProvider;
    
    #[test]
    fn it_works() {
        let thresholds = LBThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = init(thresholds, thresholds, thresholds, CPUExecutionProvider::default().into());
        cleaner.warmup(20);
        let input_img = imread("test2.jpg", IMREAD_UNCHANGED).unwrap();
        let out = cleaner.clean_mat(&input_img, CleanLevel::OriginalLB);
        if out.is_none() {
            panic!("No NSFW Content Detected");
        }
        let out = out.unwrap();
        imwrite("out.png", &out, &opencv::core::Vector::new());
    }

    #[test]
    fn folder() {
        let thresholds = LBThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = init(thresholds, thresholds, thresholds, CPUExecutionProvider::default().into());
        cleaner.warmup(20);
        let paths = std::fs::read_dir("./ai_nsfw_test").unwrap();
        let mut i = 0;
        for path in paths {
            let p = path.unwrap().path();
            let path = p.as_os_str().to_str().unwrap();
            if path.ends_with("jpg") {
                let input_img = imread(path, IMREAD_UNCHANGED).unwrap();
                let out = cleaner.clean_mat(&input_img, CleanLevel::OriginalLB);
                if out.is_none() {
                    println!("No NSFW Content Detected");
                } else {
                    let out = out.unwrap();
                    imwrite(&format!("out/out{}.png", i), &out, &opencv::core::Vector::new());
                    i+=1; 
                }
            }
        }
        println!("Finished");
    }

    #[test]
    fn test_time() {
        let thresholds = LBThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = init(thresholds, thresholds, thresholds, CPUExecutionProvider::default().into());
        cleaner.warmup(20);
        let input_img = imread("test.jpg", IMREAD_UNCHANGED).unwrap();
        let now = Instant::now();
        for _ in 0..50 {
            cleaner.clean_mat(&input_img, CleanLevel::Human);
        }
        println!("Average Time: {:?}", now.elapsed() / 50);
    }

    #[test]
    #[cfg(feature = "gif")]

    fn gif_test() {
        use std::io::Write;

        let thresholds = LBThresholds { sexy: 0.1, porn: 0.74, hentai: 0.5 };
        let cleaner = init(thresholds, thresholds, thresholds, CPUExecutionProvider::default().into());
        cleaner.warmup(20);
        let input = std::fs::File::open("unnamed.gif").unwrap();
        let out = cleaner.clean_gif(input);
        let mut file = std::fs::File::create("out.gif").unwrap();
        if out.is_none() {
            println!("No NSFW Content Detected");
        } else {
            file.write(&out.unwrap()).unwrap();
        }
    }
}
