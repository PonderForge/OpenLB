use openlb::img_filter::{ImgCleanLevel, ImgCleaner};

fn main () {
    let cleaner = ImgCleaner::builder().commit();
    cleaner.clean_file_path("test.jpg", "out.jpg", ImgCleanLevel::Human);
}