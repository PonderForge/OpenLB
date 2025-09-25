use ort::CPUExecutionProvider;
use openlb::img_filter::{ImgCleanLevel, ImgCleaner, ImgThresholds};

fn main () {
    let thresholds = ImgThresholds { sexy: 0.27, porn: 0.74, hentai: 0.5 };
    let cleaner = ImgCleaner::init(Some(thresholds), Some(thresholds), Some(CPUExecutionProvider::default().into()));
    cleaner.clean_file_path("test.jpg", "out.jpg", ImgCleanLevel::Overall);
}