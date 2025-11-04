# OpenLB
![blurred out and filtered prview of the olympic gym team](demo.webp)
OpenLB is a fast, minimal, and practical library for detecting and filtering sexual / NSFW content in images and text. It is written in Rust and designed for easy integration into existing Rust projects or services.

Features
- Image scanning with human detection and classifier models for precise filtering.
- Text scanning using tokenizer-based models for content classification.
- Small, focused API with builder patterns for easy configuration.
- Optional features to include only what you need (image_scan, text_scan, gif, bincode).
- Example usages and tests in [examples](examples), and [tests/test.rs](tests/test.rs) respectably.

Getting started

1. Requirements
- Rust (stable) toolchain.
- Place the ONNX models and tokenizer found in the latest github release into a `/model` directory next to the compiled binary.

2. Check out our examples as reference

- Image scanning (see [examples/img_filter.rs](examples/img_filter.rs) for a working example):

```rust
use openlb::img_filter::{ImgCleaner, ImgCleanLevel};

let cleaner = ImgCleaner::builder().commit();
let img = image::open("test.jpg").unwrap();
let result = cleaner.clean_image(img, ImgCleanLevel::Overall);

match result {
    None => println!("No NSFW content detected"),
    Some(filtered) => filtered.save("out.jpg").unwrap(),
}
```

Reference API: use [`openlb::img_filter::ImgCleaner`](src/img_filter/mod.rs) for image workflows.

- Text scanning (see [examples/text_filter.rs](examples/text_filter.rs) for a working example):

```rust
use openlb::text_filter::TxtCleaner;

let txt = std::fs::read_to_string("test.txt").unwrap();
let cleaner = TxtCleaner::builder().commit();
let cleaned = cleaner.clean_text(txt);
std::fs::write("out.txt", cleaned).unwrap();
```

See [`openlb::text_filter::TxtCleaner`](src/text_filter/mod.rs) for textual workflows.

3. Try out OpenLB in your own project

Why choose OpenLB
- Focused: built specifically for high-speed NSFW detection and filtration pipelines.
- Lightweight: minimal dependencies unless features are enabled.
- Practical examples and tests to get you moving quickly.

Contributing
- Bug reports and PRs are welcome. Review [tests/test.rs](tests/test.rs) and the source under [src/](src/) to learn the code structure.

AI models
- I cannot release a some of the data that trained these custom models as they were scrapped from the internet. There are some datasets that the model was trained on, that are publically available, and I will disclose them upon request.
- If the model is not accurate enough for you, contact me for support to finetune the model, do not send me the image itself if it is NSFW.

Credits
- [pykeio/ort](https://github.com/pykeio/ort): Used for all ONNX inferencing, and acceleration.
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics): Used the YOLOV11 architecture for human detection for per-human NSFW filtering.
- [Tianfang-Zhang/CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT): Used their archtecture as the backbone for our image classification system.
- [MobileBERT](https://huggingface.co/papers/2004.02984): Used their archtecture as the backbone for our textual classification system.
- Jesus Christ: My savior, my redeemer, my rock, my king, my commander, and literally the sole reason I exist. Wrote the book that warned us about lust and still loves us when we ignore it. All Hail King Jesus!


This was created by PonderForge, if you use this code, give credit where credit is due.
Pslam 111:2 "Great are the works of the LORD; they are pondered by all who delight in them."

(This README was initally written by AI as a base, however none of the code is AI generated.)