#[cfg(feature = "image_scan")]
pub mod img_filter;

#[cfg(feature = "text_scan")]
pub mod text_filter;

#[pyo3::pymodule]
mod pyopenlb {
    use pyo3::prelude::*;

    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::imgcodecs::{imwrite, imread, IMREAD_UNCHANGED};
    use ort::CPUExecutionProvider;
    use text_filter::TxtCleaner;
    use std::time::Instant;
    #[cfg(feature = "image_scan")]
    use img_filter::{ImgCleanLevel, ImgCleaner, ImgThresholds};
    
    #[test]
    #[cfg(feature = "image_scan")]
    fn clean_image() {
        let thresholds = ImgThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
        cleaner.warmup(20);
        let input_img = imread("test.jpg", IMREAD_UNCHANGED).unwrap();
        let out = cleaner.clean_mat(&input_img, ImgCleanLevel::Human);
        if out.is_none() {
            panic!("No NSFW Content Detected");
        }
        let out = out.unwrap();
        imwrite("out.png", &out, &opencv::core::Vector::new()).unwrap();
    }

    #[test]
    #[cfg(feature = "text_scan")]
    fn clean_text() {
        let txtcleaner = TxtCleaner::init(Some(0.6), Some(ort::CPUExecutionProvider::default().into()));
        txtcleaner.warmup(20);
        let now = Instant::now();
        txtcleaner.clean_string(std::fs::read_to_string("test.txt").unwrap());
        println!("Text Detect Time: {:?}", now.elapsed());
    }

    #[test]
    #[cfg(feature = "image_scan")]
    fn clean_folder() {
        let thresholds = ImgThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
        cleaner.warmup(20);
        let paths = std::fs::read_dir("./ai_nsfw_test").unwrap();
        let mut i = 0;
        for path in paths {
            let p = path.unwrap().path();
            let path = p.as_os_str().to_str().unwrap();
            if path.ends_with("jpg") {
                let input_img = imread(path, IMREAD_UNCHANGED).unwrap();
                let out = cleaner.clean_mat(&input_img, ImgCleanLevel::Human);
                if out.is_none() {
                    println!("No NSFW Content Detected");
                } else {
                    let out = out.unwrap();
                    imwrite(&format!("out/out{}.png", i), &out, &opencv::core::Vector::new()).unwrap();
                    i+=1; 
                }
            }
        }
        println!("Finished");
    }

    #[test]
    #[cfg(feature = "image_scan")]
    fn time_detection() {
        let thresholds = ImgThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
        let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
        cleaner.warmup(20);
        let input_img = imread("test.jpg", IMREAD_UNCHANGED).unwrap();
        let now = Instant::now();
        for _ in 0..50 {
            cleaner.clean_mat(&input_img, ImgCleanLevel::Human);
        }
        println!("Average Time: {:?}", now.elapsed() / 50);
    }

    #[test]
    #[cfg(feature = "gif")]
    fn clean_gif() {
        use std::io::Write;

        let thresholds = LBThresholds { sexy: 0.1, porn: 0.74, hentai: 0.5 };
        let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
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
