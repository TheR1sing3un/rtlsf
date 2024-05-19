use std::{sync::{Arc, Mutex}, time::Instant};

use crate::tlsf::{BlockHeaderPtr, InmuteableMemoryManager, MemoryManager, ThreadSafeMemoryManager, Tlsf};

/**
 * Thread-safe TLSF memory manager

    Just a wrapper of Tlsf with Arc<Mutex<Tlsf>>

 */
pub struct Tstlsf<'a> {
    tlsf: Arc<Mutex<Tlsf<'a>>>,
}

impl <'a> Tstlsf<'a> {
    pub fn new(fl_len: usize, sl_len: usize, min_block_size: usize) -> Self {
        Self {
            tlsf: Arc::new(Mutex::new(Tlsf::new(fl_len, sl_len, min_block_size))),
        }
    }
}

impl ThreadSafeMemoryManager for Tstlsf<'_> {}

impl <'a>InmuteableMemoryManager for Tstlsf<'a> {
    fn init_mem_pool(&self, mem_pool: *mut u8, mem_pool_size: usize) {
        self.tlsf.lock().unwrap().init_mem_pool(mem_pool, mem_pool_size);
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let id = std::thread::current().id();
        let timer = Instant::now();
        let block = self.tlsf.lock().unwrap().allocate(size);
        let cost = timer.elapsed();
        // println!("Tstlsf[{:?}] allocate time: {:?}", id, cost);
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        self.tlsf.lock().unwrap().deallocate(block)
    }
}
