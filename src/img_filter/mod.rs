mod human_det;
pub mod classifier;
mod box_mbr;

use std::io::Cursor;
use std::time::Instant;

use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::imageops::overlay;
use image::{ColorType, DynamicImage, GenericImage, ImageBuffer, ImageReader, Pixel, Rgb};
use ndarray::{s, ArrayBase, Dim, OwnedRepr};

use classifier::{classify_img_warmup, classify_images};
use human_det::{detect_humans_warmup, detect_humans};

#[cfg(feature = "bincode")]
use bincode::{Encode, Decode};
use ort::{ExecutionProviderDispatch, GraphOptimizationLevel, Session};

// Cut off point for NSFW images in one place
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub struct ImgThresholds {
    pub sexy: f32,
    pub porn: f32,
    pub hentai: f32,
}

impl ImgThresholds {
    pub fn new (sexy_threshold: f32, porn_threshold: f32, hentai_threshold: f32) -> ImgThresholds {
        ImgThresholds {sexy: sexy_threshold, porn: porn_threshold, hentai: hentai_threshold}
    }
}

// Image Cleaner Level
#[derive(PartialEq, Eq, Debug, Copy, Clone)]
#[cfg_attr(feature = "bincode", derive(Encode, Decode))]
pub enum ImgCleanLevel {
    Overall,
    Human
}

// Image Cleaner Builder for All the Options
pub struct ImgCleanerBuilder {
    human_thresholds: ImgThresholds,
    overall_thresholds: ImgThresholds,
    exec_provider: ExecutionProviderDispatch
}

impl ImgCleanerBuilder {
    pub fn with_human_thres (mut self, human_thres: ImgThresholds) -> ImgCleanerBuilder {
        self.human_thresholds = human_thres;
        self
    }

    pub fn with_overall_thres (mut self, overall_thres: ImgThresholds) -> ImgCleanerBuilder {
        self.overall_thresholds = overall_thres;
        self
    }

    pub fn with_exec_provider (mut self, provider: ExecutionProviderDispatch) -> ImgCleanerBuilder {
        self.exec_provider = provider;
        self
    }

    pub fn commit (self) -> ImgCleaner {
        let ort_init = ort::init()
        .with_execution_providers([self.exec_provider])
        .commit();
        if ort_init.is_err() {
            panic!("ONNX was not correctly initalized!");
        }
        //Load Models
        let detector = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../../models/human_detector.onnx")).unwrap();
        let classifier = Session::builder().unwrap().with_optimization_level(GraphOptimizationLevel::Level3).unwrap().commit_from_memory(include_bytes!("../../models/img_classifier.onnx")).unwrap();
        for _ in 0..10 {
            detect_humans_warmup(&detector);
            classify_img_warmup(&classifier);
        }
        let resize_options = ResizeOptions::new()
            .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom))
            .use_alpha(false);
        ImgCleaner { detector: detector, classifier: classifier, human_thresholds: self.human_thresholds, overall_thresholds: self.overall_thresholds, resize_options: resize_options}
    }
}

// Main Image Cleaner Struct
#[derive(Debug)]
pub struct ImgCleaner {
    detector: Session,
    classifier: Session,
    human_thresholds: ImgThresholds,
    overall_thresholds: ImgThresholds,
    resize_options: ResizeOptions
}

impl ImgCleaner {
    pub fn builder() -> ImgCleanerBuilder {
        let thresholds = ImgThresholds { sexy: 0.80, porn: 0.84, hentai: 0.80 };
        ImgCleanerBuilder {human_thresholds: thresholds, overall_thresholds: thresholds, exec_provider: ort::CPUExecutionProvider::default().into()}
    }

    pub fn clean_file_path(&self, input_path: &str, output_path: &str, level: ImgCleanLevel) {
        let out = self.clean_image(image::open(input_path).unwrap(), level);
        if out.is_some() {
            out.unwrap().save(output_path).unwrap();
        }
    }

    pub fn clean_image (&self, input_img: DynamicImage, level: ImgCleanLevel) -> Option<DynamicImage> {
        if level == ImgCleanLevel::Overall {
            let metric = classify_images(&self.classifier, &vec![input_img.clone()], &self.resize_options);
            println!("{:?}", metric);
            if metric[[0,4]] > self.human_thresholds.sexy || metric[[0,1]] > self.human_thresholds.hentai || metric[[0,3]] > self.human_thresholds.porn {
                return Some(Self::create_overlay(input_img.width(), input_img.height(), Rgb([(metric[[0,3]] * 200.0) as u8, (metric[[0,1]] * 200.0) as u8, (metric[[0,4]] * 200.0) as u8])));
            }
        } else if level == ImgCleanLevel::Human {
            let mut changed = false;
            let mut humans: Vec<(f32, f32, f32, f32, f32)> = detect_humans(&self.detector, &input_img.clone(), &self.resize_options);
            let mut human_imgs: Vec<DynamicImage> = Vec::new();
            let mut exit_img = input_img.clone();
            for human in &mut humans {
                if human.0 + human.2 > exit_img.width() as f32 {
                    human.2 += exit_img.width() as f32 - (human.0 + human.2);
                }
                if human.1 + human.3 > exit_img.height() as f32 {
                    human.3 += exit_img.height() as f32 - (human.1 + human.3);
                }
                let out = image::DynamicImage::ImageRgba8(exit_img.sub_image(human.0 as u32, human.1 as u32, human.2 as u32, human.3 as u32).to_image());
                human_imgs.push(out);
            }
            if !humans.is_empty() {
                let human_metrics = classify_images(&self.classifier, &human_imgs, &self.resize_options);
                println!("{:?}", human_metrics);
                for i in 0..humans.len() {
                    if human_metrics[[i,4]] > self.overall_thresholds.sexy || human_metrics[[i,1]] > self.overall_thresholds.hentai || human_metrics[[i,3]] > self.overall_thresholds.porn {
                        let cover = Self::create_overlay(humans[i].2 as u32, humans[i].3 as u32, Rgb([(human_metrics[[i,3]] * 200.0) as u8, (human_metrics[[i,1]] * 200.0) as u8, (human_metrics[[i,4]] * 200.0) as u8]));
                        overlay(&mut exit_img, &cover, humans[i].0 as i64, humans[i].1 as i64);
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
        return None;
    }

    pub fn classify_image(&self, input_img: DynamicImage, level: ImgCleanLevel) -> Vec<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, Option<(f32, f32, f32, f32, f32)>)> {
        let mut results: Vec<(ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, Option<(f32, f32, f32, f32, f32)>)> = Vec::new();
        if level == ImgCleanLevel::Overall {
            let metric = classify_images(&self.classifier, &vec![input_img.clone()], &self.resize_options);
            if metric[[0,4]] > self.human_thresholds.sexy || metric[[0,1]] > self.human_thresholds.hentai || metric[[0,3]] > self.human_thresholds.porn {
                results.push((metric.slice(s![0,..]).to_owned(), None));
            }
        } else if level == ImgCleanLevel::Human {
            let humans: Vec<(f32, f32, f32, f32, f32)> = detect_humans(&self.detector, &input_img.clone(), &self.resize_options);
            let mut human_imgs: Vec<DynamicImage> = Vec::new();
            let mut exit_img = input_img.clone();
            for human in &humans {
                let out = image::DynamicImage::ImageRgba8(exit_img.sub_image(human.0 as u32, human.1 as u32, human.2 as u32, human.3 as u32).to_image());
                human_imgs.push(out);
            }
            if !humans.is_empty() {
                let human_metrics = classify_images(&self.classifier, &human_imgs, &self.resize_options);
                for i in 0..humans.len() {
                    results.push((human_metrics.slice(s![i,..]).to_owned(), Some(humans[i])));
                }
            }
        }
        results
    }

    fn create_overlay (width: u32, height: u32, color: Rgb<u8>) -> DynamicImage {
        let mut husk = ImageBuffer::from_pixel(width, height, color.to_rgba());
        let icon = ImageReader::new(Cursor::new(include_bytes!("../../icon.tiff"))).with_guessed_format().unwrap().decode().unwrap();
        let mut resizer = Resizer::new();
        let mut reicon = if height < width {
            DynamicImage::new(height, height, ColorType::Rgba8)
        } else {
            DynamicImage::new(width, width, ColorType::Rgba8)
        };
        
        resizer.resize(&icon, &mut reicon, Some(&ResizeOptions::new().resize_alg(ResizeAlg::Nearest))).unwrap();
        if height < width {
            overlay(&mut husk, &reicon, ((width-height)/2) as i64, 0);
        } else {
            overlay(&mut husk, &reicon, 0, ((height-width)/2) as i64);
        }
        husk.into()
    }

    #[cfg(feature = "gif")]
    pub fn clean_gif_nd<R> (&self, input_gif: R) -> Option<Vec<u8>> where R: std::io::Read {

        use image::Rgba;
        use gif::Frame;

        let mut decoder = gif::DecodeOptions::new();
        decoder.set_color_output(gif::ColorOutput::RGBA);
        let decoder = decoder.read_info(input_gif).unwrap();
        let width = decoder.width() as u32;
        let height = decoder.height() as u32;
        let colormap = decoder.global_palette().unwrap_or(&[]).to_owned();
        let repeat = decoder.repeat();
        let mut cleaned_frames: Vec<Frame> = Vec::new();
        let mut decoder_iter = decoder.into_iter();
        let mut last: ImageBuffer<Rgba<u8>, std::vec::Vec<u8>> = ImageBuffer::from_raw(width, height, decoder_iter.next().unwrap().unwrap().buffer.clone().into_owned()).unwrap();
        while let Some(Ok(frame)) = decoder_iter.next() {
            let data = &frame.buffer;
            let image: ImageBuffer<Rgba<u8>, std::vec::Vec<u8>> = ImageBuffer::from_raw(frame.width as u32, frame.height as u32, data.clone().into_owned()).unwrap();
            overlay(&mut last, &image, frame.left as i64, frame.top as i64);
            let filtered = self.clean_image(DynamicImage::ImageRgba8(last.clone()), ImgCleanLevel::Human);
            let mut newframe: Frame = if filtered.is_some() {
                Frame::from_rgb_speed(width as u16, height as u16, &mut filtered.unwrap().into_rgb8().to_vec(), 15)
            } else {
                Frame::from_rgb_speed(width as u16, height as u16, &mut DynamicImage::ImageRgba8(last.clone()).into_rgb8().to_vec(), 15)
            };
            newframe.delay = frame.delay;
            cleaned_frames.push(newframe);
        }
        let mut out_file: Vec<u8> = Vec::new();
        {
            let mut encoder = gif::Encoder::new(&mut out_file, width as u16, height as u16, &colormap).unwrap();
            encoder.set_repeat(repeat).unwrap();
            for state in &cleaned_frames {
                encoder.write_frame(&state).unwrap();
            }
        }
        return Some(out_file);
    }
}
