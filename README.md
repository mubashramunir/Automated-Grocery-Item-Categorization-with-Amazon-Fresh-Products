# Automated Product Categorization with YOLOv8 — Amazon Fresh Case Study

An end-to-end computer vision pipeline that detects and categorizes retail products from images using a custom-trained YOLOv8 object detection model — built to explore how automated visual categorization could reduce manual tagging effort in e-commerce catalog management.

## Overview

Large retail catalogs rely heavily on manual product tagging, which is slow and error-prone at scale. This project tests whether a custom object detection model can automatically identify and categorize products from images alone, using Amazon Fresh's product catalog as a case study.

The pipeline covers the full lifecycle: **scraping training images → building a labeled dataset → training a YOLOv8 model → validating predictions → deploying an interactive demo.**

## What it does

- **Custom data collection** — an automated Playwright scraper pulls product images directly from Amazon Fresh category pages, handling dynamic page scrolling and lazy-loaded images
- **46-class object detection model** — trained with YOLOv8 across a broad product catalog spanning fresh produce, meat & seafood, dairy, and personal care items (e.g. apples, chicken, seafood, cereals, shampoo, foundation)
- **Iterative hyperparameter tuning** — multiple training passes adjusting epochs, learning rate, optimizer (AdamW), weight decay, and warmup schedule to improve detection accuracy
- **Human-in-the-loop validation tool** — a custom review widget steps through predictions image-by-image so results can be manually marked correct/incorrect, supporting a feedback loop for model improvement
- **Model export & evaluation** — trained model exported to ONNX format and benchmarked with per-class mAP scores
- **Interactive demo** — a Gradio web app where a user uploads any image and receives detected objects with bounding boxes and category labels in real time

## Tech stack

`Python` · `YOLOv8 (Ultralytics)` · `OpenCV` · `Playwright` (web scraping) · `Gradio` (demo UI) · `ONNX` (model export) · Google Colab / Drive (training environment)

## Pipeline

1. **Scrape** — `scrape_images()` collects product images from Amazon Fresh category pages
2. **Configure** — dataset paths and 46 class names are written to a `dataset.yaml` file for YOLO training
3. **Train** — YOLOv8n is fine-tuned on the custom dataset, with a second pass using tuned hyperparameters (AdamW, weight decay, warmup)
4. **Validate** — the `ModelReviewer` widget displays predictions on held-out images for manual correct/incorrect labeling
5. **Evaluate** — per-class mAP is calculated from validation results
6. **Export** — model is exported to ONNX for lighter-weight inference
7. **Deploy** — a Gradio interface exposes the model for live image upload and detection

## Results

*(Add your actual validation numbers here, e.g.: "Achieved X% mAP@0.5 across 46 classes after hyperparameter tuning" — pull this from your `results.results_dict` output.)*

## Try it

The Gradio app (`interface.launch()`) allows anyone to upload a product image and get back detected categories with bounding boxes — a working demo of the model rather than just training code.

## Notes / limitations

- Trained on a relatively small custom dataset scraped from a single source; performance would need validation against a larger, more balanced dataset for production use
- Class list mixes grocery and personal-care items, reflecting Amazon Fresh's actual catalog breadth rather than a narrow "groceries only" scope
