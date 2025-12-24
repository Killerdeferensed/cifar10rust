
use burn::{data::{dataloader::batcher::Batcher, dataset::Dataset}, prelude::*};
use cifar_10_loader::{CifarDataset, CifarImage};

#[derive(Clone,Debug)]
pub struct CifarImageRaw{
    pub pixels: [u8; 32*32*3],
    pub label: u8
}

impl CifarImageRaw{
    pub fn from(image: &CifarImage) -> Self{
        Self { pixels: image.image.raw_pixels().try_into().expect("MismatchedSize"), label: image.label }
    }
}
pub struct CifarDatasetWrapper{
    dataset: Vec<CifarImageRaw>,
}

impl CifarDatasetWrapper {
    pub fn train(path: &str) -> Self {
        let set = CifarDataset::new(path).unwrap().train_dataset;
        let set = set.iter().map(|item| CifarImageRaw::from(item)).collect();
        Self {dataset: set}
    }
    pub fn test(path: &str) -> Self {
        let set = CifarDataset::new(path).unwrap().test_dataset;
        let set = set.iter().map(|item| CifarImageRaw::from(item)).collect();
        Self {dataset: set}
    }
}

impl Dataset<CifarImageRaw> for CifarDatasetWrapper{
    fn get(&self, index: usize) -> Option<CifarImageRaw> {
        self.dataset.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone)]
pub struct CifarBatcher<B: Backend>{
    device: B::Device,
}

impl<B: Backend> CifarBatcher<B>{
    pub fn new(device: B::Device) -> Self{
        Self {device}
    }
}
#[derive(Clone,Debug)]
pub struct CifarBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<CifarImageRaw, CifarBatch<B>> for CifarBatcher<B> {
    fn batch(&self, items: Vec<CifarImageRaw>) -> CifarBatch<B> {
        let images = items
            .iter()
            // .map(|item| TensorData::from(item.pixels.as_slice()).convert::<B::FloatElem>())  
            .map(|data| Tensor::<B,1>::from_data(data.pixels.as_slice(), &self.device))
            .map(|tensor| tensor.reshape([1,32,32,3]))
            .map(|tensor| tensor/255)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    [(item.label as i64).elem::<B::IntElem>()]
                    , &self.device
                )
            })
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        CifarBatch{images, targets}
    }
}
