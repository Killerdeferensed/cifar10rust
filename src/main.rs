mod model;
mod data;
mod training;
mod inference;

use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu};
use data::CifarDatasetWrapper;
use model::ModelConfig;
use training::TrainingConfig;

fn main() {
    type CifarBackend = Wgpu<f32,i32>;
    type CifarAutodiffBackend = Autodiff<CifarBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./out/cifar";

    training::train::<CifarAutodiffBackend>(
        artifact_dir, 
        TrainingConfig::new(ModelConfig::new(10, 128).with_dropout(0.15), AdamConfig::new()).with_num_workers(8).with_batch_size(48),
         device.clone(),
        );

    inference::infer::<CifarBackend>(
        artifact_dir, 
        device, 
        CifarDatasetWrapper::test("./cifar-10-batches-bin").get(42).unwrap(),
    );
}
