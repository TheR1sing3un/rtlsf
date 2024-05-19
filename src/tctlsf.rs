use std::{cell::RefCell, rc::Rc, sync::{atomic::AtomicU64, Arc, Mutex}, thread::{self, ThreadId}, time::Instant};
use std::sync::atomic::Ordering;

use hashbrown::HashSet;
use log::{debug, info};
use crate::tlsf::{BlockHeader, BlockHeaderPtr, InmuteableMemoryManager, MemoryManager, ThreadSafeMemoryManager, Tlsf};


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
        self.cache_hit_count.fetch_add(count, Ordering::SeqCst);
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
        let thread_id = curr_thread.id();
        let block = THREAD_LOCAL_A.with(|pool| {
            // Allocate from thread memory pool
            let mut block = pool.borrow().allocate(size);
            if block.is_some() {
                self.cache_hit_add(1);
                unsafe {
                    info!("Tctlsf[{:?}]: Allocate from cache, allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", thread_id, size, block.unwrap().as_ref().size(), timer.elapsed().as_nanos());
                }
                return block;
            }
            // Allocate from core memory pool
            let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
            block = core_mem_pool.allocate(size);
            if block.is_none() {
                return None;
            }

            // Add this block to thread memory pool for hold
            pool.borrow().hold_block(block.unwrap());
            
            unsafe {
                info!("Tctlsf[{:?}]: Allocate from core, allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", thread_id, size, block.unwrap().as_ref().size(), timer.elapsed().as_nanos());
            }
            block
        });
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        let curr_thread = thread::current();
        let thread_id = curr_thread.id();

        let blcok = THREAD_LOCAL_A.with(|pool| {
            let new_block = pool.borrow().deallocate(block);

            // After deallocation, we should check if we should take back some blocks from thread memory pool to core memory pool
            if pool.borrow().dehold_check() {
                // Dehold it from thread memory pool
                pool.borrow().dehold(new_block);
                // Deallocate it from core memory pool
                {
                    let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
                    // TODO: Deal with core pool merge the blocks which are holded by thread memory pool
                    core_mem_pool.deallocate(new_block);
                }
            }

            block
        });

        blcok
    }

}

struct BlockMergeHelper {
    id: ThreadId,
    block_set: RefCell<HashSet<BlockHeaderPtr>>,
    
    /* Free blocks' total size and total num record */ 
    free_blocks_size: RefCell<usize>,
    free_blocks_num: RefCell<usize>,

     /* Holded blocks' total size and total num record */
    holded_blocks_size: RefCell<usize>,
    holded_blocks_num: RefCell<usize>,
}

impl BlockMergeHelper {
    pub fn new(id: ThreadId) -> Self {
        Self {
            id,
            block_set: RefCell::new(HashSet::new()),
            free_blocks_size: RefCell::new(0),
            free_blocks_num: RefCell::new(0),
            holded_blocks_size: RefCell::new(0),
            holded_blocks_num: RefCell::new(0),
        }
    }

    // Only merge the block that this block is holded by this thread cache memory pool
    // And the block should be valid and free
    fn merge_permit(&self, block: BlockHeaderPtr) -> bool {
        unsafe {
            self.block_set.borrow().contains(&block) && block.as_ref().is_valid() && block.as_ref().is_free()
        }
    }

    fn insert_block_to_set(&self, block: BlockHeaderPtr) {
        self.block_set.borrow_mut().insert(block);
    }

    fn remove_block_from_set(&self, block: &BlockHeaderPtr) {
        self.block_set.borrow_mut().remove(block);
    }

    fn hold_block(&self, block: BlockHeaderPtr) {
        unsafe {
            let size = block.as_ref().size();
            *self.holded_blocks_size.borrow_mut() += size;
            *self.holded_blocks_num.borrow_mut() += 1;
            self.insert_block_to_set(block);
            debug!("BlockMergeHelper[{:?}]: Hold a block[{:?}], size: {:?}B", self.id, block.as_ptr() as usize, size);
        }
    }

    fn add_free_block(&self, num: usize, size: usize) {
        *self.free_blocks_size.borrow_mut() += size;
        *self.free_blocks_num.borrow_mut() += num;
    }

    fn remove_free_block(&self, num: usize, size: usize) {
        *self.free_blocks_size.borrow_mut() -= size;
        *self.free_blocks_num.borrow_mut() -= num;
    }
     fn remove_merged_block(&self, size: usize) {
        *self.free_blocks_size.borrow_mut() -= size;
        *self.free_blocks_num.borrow_mut() -= 1;
        *self.holded_blocks_num.borrow_mut() -= 1;
    }
    fn dehold_block(&self, block: BlockHeaderPtr) {
        unsafe {
            let size = block.as_ref().size();
            *self.holded_blocks_size.borrow_mut() -= size;
            *self.holded_blocks_num.borrow_mut() -= 1;
            *self.free_blocks_num.borrow_mut() -= 1;
            *self.free_blocks_size.borrow_mut() -= size;
            self.remove_block_from_set(&block);
            debug!("BlockMergeHelper[{:?}]: Dehold a block[{:?}], size: {:?}B", self.id, block.as_ptr() as usize, size);
        }
    }

    fn free_blocks_size(&self) -> usize {
        *self.free_blocks_size.borrow()
    }

    fn free_blocks_num(&self) -> usize {
        *self.free_blocks_num.borrow()
    }

    fn holded_blocks_size(&self) -> usize {
        *self.holded_blocks_size.borrow()
    }

    fn holded_blocks_num(&self) -> usize {
        *self.holded_blocks_num.borrow()
    }

}

pub struct ThreadCacheMemPool<'a> {
    thread_id: ThreadId,

    /* Tlsf parameter */
    max_size: usize,
    fl_len: usize,
    sl_len: usize,
    min_block_size: usize,

    tlsf: RefCell<Tlsf<'a>>,
    /* Helper helps cache to determine if deallocated block's neighbor should be merged */
    block_merge_helper: Rc<BlockMergeHelper>,
}

impl <'a>ThreadCacheMemPool<'a> {
    pub fn new(max_size: usize, fl_len: usize, sl_len: usize, min_block_size: usize) -> Self {
        let mut tlsf = Tlsf::new(fl_len, sl_len, min_block_size);
        let thread_id = thread::current().id();
        let helper = Rc::new(BlockMergeHelper::new(thread_id));
        let clone_helper = helper.clone();

        // Register merge function, only merge the block that this block is holded by this thread cache memory pool
        tlsf.register_merge_func(Box::new(move |block| {
            clone_helper.merge_permit(block)
        }));

        let pool = Self {
            thread_id,
            max_size,
            fl_len,
            sl_len,
            min_block_size,
            tlsf: RefCell::new(tlsf),
            block_merge_helper: helper,
        };
        pool
    }

    fn hold_block(&self, block: BlockHeaderPtr) {
        self.block_merge_helper.hold_block(block);
    }

    fn insert_block_to_set(&self, block: BlockHeaderPtr) {
        self.block_merge_helper.insert_block_to_set(block);
    }

    fn remove_block_from_set(&self, block: &BlockHeaderPtr) {
        self.block_merge_helper.remove_block_from_set(block);
    }
    /**
     * Determine if we should dehold some blocks from thread memory pool to core memory pool
     * (1) If the free blocks' total size is larger than thread memory pool's max size
     */
    fn dehold_check(&self) -> bool {
        self.block_merge_helper.holded_blocks_size() > self.max_size
    }

    fn dehold(&self, block: BlockHeaderPtr) {
        // Remove it from tlsf
        self.tlsf.borrow_mut().remove_free_block(block);
        // Dehold it
        self.block_merge_helper.dehold_block(block);
    }
    
}


impl <'a>InmuteableMemoryManager for ThreadCacheMemPool<'a> {

    fn init_mem_pool(&self, _mem_pool: *mut u8, _mem_pool_size: usize) {
        panic!("ThreadCacheMemPool should not call init_mem_pool");
    }

    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr> {
        let timer = Instant::now();
        let mut tlsf = self.tlsf.borrow_mut();
        // TODO: Deal with splited block's index in cache
        let block = tlsf.allocate(size);
        let cost = timer.elapsed().as_nanos();
        if let Some(b) = block {
            // Update free blocks' total size and total num record
            unsafe {
                self.block_merge_helper.remove_free_block(1, b.as_ref().size());
            }
            unsafe {
                info!("ThreadCache[{:?}]: Allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", self.thread_id, size, b.as_ref().size(), cost);
            }
        } else {
            info!("ThreadCache[{:?}]: Allocate min size: {:?}B, real block's size: None, cost time: {:?}", self.thread_id, size, cost);
        }
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        unsafe {
            let timer = Instant::now();
            let address = block.as_ptr() as usize;
            let next_block = block.as_ref().next_block();
            let prev_block = block.as_ref().prev_phys_block;
            let new_block = self.tlsf.borrow_mut().deallocate(block);
            if new_block != block {
                // The block has merged with previous block
                // Remove deallocated block from block_set
                self.remove_block_from_set(&block);
                // Remove prev block from free blocks' total size and total num record
                self.block_merge_helper.remove_merged_block(prev_block.unwrap().as_ref().size());

            }

            if next_block.is_some() && next_block != new_block.as_ref().next_block() {
                // The block has merged with next block
                // Remove original next block from block_set
                self.remove_block_from_set(&next_block.unwrap());
                // Remove original next block from free blocks' total size and total num record
                self.block_merge_helper.remove_merged_block(next_block.unwrap().as_ref().size());
            }

            // Add new block to block_set
            self.insert_block_to_set(new_block);
            self.block_merge_helper.add_free_block(1, new_block.as_ref().size());
            let cost = timer.elapsed().as_nanos();
            info!("ThreadCache[{:?}]: Deallocate block[{:?}], new block[{:?}], cost time: {:?}", self.thread_id, address, new_block.as_ptr() as usize, cost);

            new_block
        }
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