pub trait Bitmap {
    fn new (len: usize) -> Self;
    fn set(&mut self, index: usize);
    fn clear(&mut self, index: usize);
    fn get(&self, index: usize) -> bool;
    fn len(&self) -> usize;
}

pub struct DefaultBitmap {
    len: usize,
    bitmap: Vec<u8>,
}

impl Bitmap for DefaultBitmap {

    fn new(len: usize) -> Self {
        Self {
            len,
            bitmap: vec![0; len / 8 + 1],
        }
    }

    fn set(&mut self, index: usize) {
        self.bitmap[index / 8] |= 1 << (index % 8);
    }

    fn clear(&mut self, index: usize) {
        self.bitmap[index / 8] &= !(1 << (index % 8));
    }

    fn get(&self, index: usize) -> bool {
        self.bitmap[index / 8] & (1 << (index % 8)) != 0
    }

    fn len(&self) -> usize {
        self.len
    }
}