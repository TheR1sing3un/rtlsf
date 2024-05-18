use std::{borrow::BorrowMut, cell::RefCell, iter::Once, sync::{atomic::AtomicU64, Arc, Mutex}, thread::{self, sleep, ThreadId}, time::{Duration, Instant}};
use std::sync::atomic::Ordering;
use dashmap::DashMap;
use hashbrown::{HashMap, HashSet};
use crate::tlsf::{self, BlockHeaderPtr, InmuteableMemoryManager, MemoryManager, ThreadSafeMemoryManager, Tlsf};


const DEFAULT_TMP_MAX_SIZE: usize = 1 << 20;
const DEFAULT_TMP_FL_LEN: usize = 26;
const DEFAULT_TMP_SL_LEN: usize = 4;
const DEFAULT_TMP_MIN_BLOCK_SIZE: usize = 64;

const DEFAULT_CMP_MAX_SIZE: usize = 1 << 20;
const DEFAULT_CMP_FL_LEN: usize = 26;
const DEFAULT_CMP_SL_LEN: usize = 4;
const DEFAULT_CMP_MIN_BLOCK_SIZE: usize = 64;


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
    // thread_mem_pools: Arc<Mutex<HashMap<ThreadId/* thread id */, Arc<ThreadCacheMemPool<'a>>>/* thread-cached memory-pool */>>,

    cache_hit_count: AtomicU64,
    cache_miss_count: AtomicU64,
    
}

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
            // thread_mem_pools: Arc::new(Mutex::new(HashMap::new())),
            cache_hit_count: AtomicU64::new(0),
            cache_miss_count: AtomicU64::new(0),   
        }
    }

    fn cache_hit_count(&self) -> u64 {
        self.cache_hit_count.load(Ordering::SeqCst)
    }

    fn cache_miss_count(&self) -> u64 {
        self.cache_miss_count.load(Ordering::SeqCst)
    }

    fn cache_hit_add(&self, count: u64) {
        let timer = Instant::now();
        self.cache_hit_count.fetch_add(count, Ordering::SeqCst);
        // println!("Cache hit add cost: {}ns", timer.elapsed().as_nanos());
    }

    fn cache_miss_add(&self, count: u64) {
        self.cache_miss_count.fetch_add(count, Ordering::SeqCst);
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hit_count = self.cache_hit_count();
        let miss_count = self.cache_miss_count();
        if hit_count + miss_count == 0 {
            return 0.0;
        }
        hit_count as f64 * 100.0 / (hit_count + miss_count) as f64
    }


}

unsafe impl <'a>Send for Tctlsf<'a> {}
unsafe impl <'a>Sync for Tctlsf<'a> {}

thread_local! {
    static THREAD_LOCAL_A: RefCell<ThreadCacheMemPool<'static>> = RefCell::new(ThreadCacheMemPool::new( DEFAULT_TMP_MAX_SIZE, DEFAULT_TMP_FL_LEN, DEFAULT_TMP_SL_LEN, DEFAULT_TMP_MIN_BLOCK_SIZE));
}

impl ThreadSafeMemoryManager for Tctlsf<'_> {}

impl <'a>InmuteableMemoryManager for Tctlsf<'a> {
    
    fn init_mem_pool(&self, mem_pool: *mut u8, mem_pool_size: usize) {
        let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
        core_mem_pool.init_mem_pool(mem_pool, mem_pool_size);
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let curr_thread = thread::current();
        let timer = Instant::now();
        // println!("[[[[[[[[[[[[[[TC-Tlsf[{:?}]: Allocate from thread cache memory pool start", curr_thread.id());
        let thread_id = curr_thread.id();
        let block = THREAD_LOCAL_A.with(|pool| {
            // Allocate from thread memory pool
            let mut block = pool.borrow().allocate(size);
            if block.is_some() {
                self.cache_hit_add(1);
                // println!("*********TC-Tlsf[{:?}]: Cache hit, cost: {}ns", thread_id, timer.elapsed().as_nanos());
                return block;
            }
            // Allocate from core memory pool
            let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
            block = core_mem_pool.allocate(size);
            if block.is_none() {
                return None;
            }

            // Add block to block_set
            pool.borrow().add_block_to_set(block.unwrap());
            // println!("]]]]]]]]]]]]TC-Tlsf[{:?}]: Cache miss, cost: {}", thread_id, timer.elapsed().as_nanos());
            block
        });
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        let curr_thread = thread::current();
        let thread_id = curr_thread.id();

        THREAD_LOCAL_A.with(|pool| {
            let new_block = pool.borrow().deallocate(block);
            // TODO: When we back the blocks from thread memory pool to core memory pool?
            new_block
        })
    }

    fn print_metrics(&self) {
        // println!("Cache hit rate: {}", self.cache_hit_rate());
    }

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

pub struct ThreadCacheMemPool<'a> {
    thread_id: ThreadId,

    /* Tlsf parameter */
    max_size: usize,
    fl_len: usize,
    sl_len: usize,
    min_block_size: usize,

    tlsf: RefCell<Tlsf<'a>>,
    helper: Arc<Mutex<Helper>>,

    /* Holded blocks' total size and total num record */
    holded_blocks_size: usize,
    holded_blocks_num: usize,

    /* Free blocks' total size and total num record */
    free_blocks_size: usize,
    free_blocks_num: usize,
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
        let thread_id = thread::current().id();
        let pool = Self {
            thread_id,
            max_size,
            fl_len,
            sl_len,
            min_block_size,
            tlsf: RefCell::new(tlsf),
            helper: wrap_helper,
            holded_blocks_size: 0,
            holded_blocks_num: 0,
            free_blocks_size: 0,
            free_blocks_num: 0,
        };
        pool
    }

    pub fn add_block_to_set(&self, block: BlockHeaderPtr) {
        self.helper.lock().unwrap().block_set.insert(block);
    }
    
}


impl <'a>InmuteableMemoryManager for ThreadCacheMemPool<'a> {

    fn init_mem_pool(&self, _mem_pool: *mut u8, _mem_pool_size: usize) {
        panic!("ThreadCacheMemPool should not call init_mem_pool");
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let timer = Instant::now();
        // println!("<<<<<<<<<<<<<<ThreadCache[{:?}]: Allocate from thread memory pool start", self.thread_id);
        let mut tlsf = self.tlsf.borrow_mut();
        // println!("*********ThreadCache[{:?}]: Get cache lock cost: {}", self.thread_id, timer.elapsed().as_nanos());
        let block = tlsf.allocate(size);
        // println!(">>>>>>>>>>>>>>ThreadCache[{:?}]: Allocate {:?}, cost: {}", self.thread_id, block.is_some(), timer.elapsed().as_nanos());
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        unsafe {
            let next_block = block.as_ref().next_block();
            let new_block = self.tlsf.borrow_mut().deallocate(block);
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

    fn print_metrics(&self) {
        println!("Holded blocks size: {}, Holded blocks num: {}", self.holded_blocks_size, self.holded_blocks_num);
        println!("Free blocks size: {}, Free blocks num: {}", self.free_blocks_size, self.free_blocks_num);
    }
}

#[cfg(test)]
mod tests {
    use super::Tctlsf;
    use super::DEFAULT_CMP_FL_LEN;
    use super::DEFAULT_CMP_MAX_SIZE;
    use super::DEFAULT_CMP_MIN_BLOCK_SIZE;
    use super::DEFAULT_CMP_SL_LEN;
    use super::DEFAULT_TMP_FL_LEN;
    use super::DEFAULT_TMP_MAX_SIZE;
    use super::DEFAULT_TMP_MIN_BLOCK_SIZE;
    use super::DEFAULT_TMP_SL_LEN;
    use crate::tlsf::InmuteableMemoryManager;
    use crate::tlsf::{
        ThreadSafeMemoryManager,
        BlockHeader,
    };

    #[test]
    fn test_basical_alloc_dealloc() {
        let dsa = Tctlsf::new(
            DEFAULT_TMP_MAX_SIZE, DEFAULT_TMP_FL_LEN, DEFAULT_TMP_SL_LEN, DEFAULT_TMP_MIN_BLOCK_SIZE,
            DEFAULT_CMP_MAX_SIZE, DEFAULT_CMP_FL_LEN, DEFAULT_CMP_SL_LEN, DEFAULT_CMP_MIN_BLOCK_SIZE
        );
        unsafe {
            let mut arena: [std::mem::MaybeUninit<u8>; 1 << 18] = std::mem::MaybeUninit::uninit().assume_init();
            dsa.init_mem_pool(arena.as_mut_ptr() as *mut u8, arena.len());

            // Allocate a block with size 64
            let block = dsa.allocate(64).unwrap();

            // Allocate a block with size 128
            let block2 = dsa.allocate(128).unwrap();

            assert_eq!(block.as_ref().next_block(), Some(block2));

            // Deallocate block 1
            dsa.deallocate(block);

            // Allocate a block with size 128
            let block3 = dsa.allocate(128).unwrap();

            assert_eq!(block2.as_ref().next_block().unwrap(), block3);

            // Deallocate block 3
            dsa.deallocate(block3);

            // Allocate a block with 144
            let block4 = dsa.allocate(144).unwrap();

            assert_eq!(block3.as_ref().next_block().unwrap(), block4);

            // Deallocate block 2
            dsa.deallocate(block2);

            // Allocate a block with 320
            let block5 = dsa.allocate(320).unwrap();

            assert_eq!(block5.as_ref().next_block().unwrap(), block4);
            assert_eq!(block5, block);   
        }
    }
}