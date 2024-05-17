use std::{borrow::BorrowMut, sync::{Arc, Mutex}, thread::{self, ThreadId}};

use hashbrown::{HashMap, HashSet};
use crate::tlsf::{BlockHeaderPtr, MemoryManager, ThreadSafeMemoryManager, Tlsf};

pub struct Tctlsf<'a> {

    /* Memory pool parameter */

    /************** Thread memory pool parameter **************/
    tmp_max_size: usize, // Maximum size of thread memory pool
    tmp_fl_len: usize, // First-Level bitmap length of thread memory pool
    tmp_sl_len: usize, // Second-Level bitmap length of thread memory pool
    tmp_min_block_size: usize, // Minimum block size of thread memory pool
    /************** Thread memory pool parameter **************/

    /************** Core memory pool parameter **************/ 
    cmp_max_size: usize, // Maximum size of core memory pool
    cmp_fl_len: usize, // First-Level bitmap length of core memory pool
    cmp_sl_len: usize, // Second-Level bitmap length of core memory pool
    cmp_min_block_size: usize, // Minimum block size of core memory pool
    /************** Core memory pool parameter **************/


    core_mem_pool: Arc<Mutex<Tlsf<'a>>>,
    thread_mem_pools: Arc<Mutex<HashMap<ThreadId/* thread id */, ThreadCacheMemPool<'a>>/* thread-cached memory-pool */>>,
}

pub struct ThreadCacheMemPool<'a> {
    max_size: usize,
    fl_len: usize,
    sl_len: usize,
    min_block_size: usize,
    tlsf: Arc<Mutex<Tlsf<'a>>>,
    helper: Arc<Mutex<Helper>>,
}

struct Helper {
    block_set: HashSet<BlockHeaderPtr>,
}

impl Helper {

    pub fn new(block_set: HashSet<BlockHeaderPtr>) -> Self {
        Self {
            block_set,
        }
    }

    // Only merge the block that this block is holded by this thread cache memory pool
    // And the block should be valid and free
    pub fn merge_permit(&self, block: BlockHeaderPtr) -> bool {
        unsafe {
            self.block_set.contains(&block) && block.as_ref().is_valid() && block.as_ref().is_free()
        }
    }
}

unsafe impl Send for Helper {}
unsafe impl Sync for Helper {}

impl <'a>Tctlsf<'a> {
    pub fn new(
        tmp_max_size: usize, tmp_fl_len: usize, tmp_sl_len: usize, tmp_min_block_size: usize,
        cmp_max_size: usize, cmp_fl_len: usize, cmp_sl_len: usize, cmp_min_block_size: usize
    ) -> Self {
        Self {
            tmp_max_size,
            tmp_fl_len,
            tmp_sl_len,
            tmp_min_block_size,
            cmp_max_size,
            cmp_fl_len,
            cmp_sl_len,
            cmp_min_block_size,
            core_mem_pool: Arc::new(Mutex::new(Tlsf::new(cmp_fl_len, cmp_sl_len, cmp_min_block_size))),
            thread_mem_pools: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl <'a>ThreadSafeMemoryManager for Tctlsf<'a> {
    
    fn init_mem_pool(&self, mem_pool: *mut u8, mem_pool_size: usize) {
        let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
        core_mem_pool.init_mem_pool(mem_pool, mem_pool_size);
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let curr_thread = thread::current();
        let thread_id = curr_thread.id();
        // TODO: More efficient Thread-Safe HashMap
        let mut thread_mem_pools = self.thread_mem_pools.lock().unwrap();

        // Get or insert thread memory pool
        let thread_mem_pool = thread_mem_pools.entry(thread_id).or_insert_with(|| {
            println!("Create new thread memory pool for thread {:?}", thread_id);
            ThreadCacheMemPool::new(self.tmp_max_size, self.tmp_fl_len, self.tmp_sl_len, self.tmp_min_block_size)
        });

        // Allocate from thread memory pool
        let mut block = thread_mem_pool.allocate(size);
        if block.is_some() {
            return block;
        }

        // Allocate from core memory pool
        let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
        block = core_mem_pool.allocate(size);
        if block.is_none() {
            return None;
        }

        // Add block to block_set
        thread_mem_pool.add_block_to_set(block.unwrap());

        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        let curr_thread = thread::current();
        let thread_id = curr_thread.id();
        let mut thread_mem_pools = self.thread_mem_pools.lock().unwrap();
        let thread_mem_pool = thread_mem_pools.get_mut(&thread_id).unwrap();
        let new_block = thread_mem_pool.deallocate(block);
        // TODO: When we back the blocks from thread memory pool to core memory pool?
        new_block
    }

}

impl <'a>ThreadCacheMemPool<'a> {
    pub fn new(max_size: usize, fl_len: usize, sl_len: usize, min_block_size: usize) -> Self {
        let mut tlsf = Tlsf::new(fl_len, sl_len, min_block_size);
        let helper = Helper::new(HashSet::new());
        let wrap_helper = Arc::new(Mutex::new(helper));
        let cloned_func = wrap_helper.clone();
        tlsf.register_merge_func(Box::new(move |block | {
            cloned_func.lock().unwrap().merge_permit(block)
        }));
        let pool = Self {
            max_size,
            fl_len,
            sl_len,
            min_block_size,
            tlsf: Arc::new(Mutex::new(tlsf)),
            helper: wrap_helper,
        };
        pool
    }

    pub fn add_block_to_set(&mut self, block: BlockHeaderPtr) {
        self.helper.lock().unwrap().block_set.insert(block);
    }
    
}


impl <'a>ThreadSafeMemoryManager for ThreadCacheMemPool<'a> {

    fn init_mem_pool(&self, _mem_pool: *mut u8, _mem_pool_size: usize) {
        panic!("ThreadCacheMemPool should not call init_mem_pool");
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let block = self.tlsf.lock().unwrap().borrow_mut().allocate(size);
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        unsafe {
            let next_block = block.as_ref().next_block();
            let new_block = self.tlsf.lock().unwrap().borrow_mut().deallocate(block);
            let mut helper = self.helper.lock().unwrap();
            if new_block != block {
                // The block has merged with previous block
                // Remove deallocated block from block_set
                helper.block_set.remove(&block);
            }

            if next_block.is_some() && next_block != new_block.as_ref().next_block() {
                // The block has merged with next block
                // Remove original next block from block_set
                helper.block_set.remove(&next_block.unwrap());
            }

            // Add new block to block_set
            helper.block_set.insert(new_block);
            new_block
        }
    }
}