use std::{any::Any, borrow::Borrow, cell::RefCell, num::NonZeroU64, rc::Rc, sync::{atomic::AtomicU64, Arc, Mutex}, thread::{self, ThreadId}, time::Instant};
use std::sync::atomic::Ordering;

use hashbrown::HashSet;
use log::{debug, info};
use crate::tlsf::{BlockHeaderPtr, InmuteableMemoryManager, MemoryManager, ThreadSafeMemoryManager, Tlsf};


const DEFAULT_TMP_MAX_SIZE: usize = 1 << 20;
const DEFAULT_TMP_FL_LEN: usize = 26;
const DEFAULT_TMP_SL_LEN: usize = 4;
const DEFAULT_TMP_MIN_BLOCK_SIZE: usize = 64;

const DEFAULT_CMP_MAX_SIZE: usize = 1 << 20;
const DEFAULT_CMP_FL_LEN: usize = 26;
const DEFAULT_CMP_SL_LEN: usize = 4;
const DEFAULT_CMP_MIN_BLOCK_SIZE: usize = 64;

const DEFAULT_CORE_TLSF_ID: usize = 131313;
const DEFAULT_UNHOLDED_ID: usize = 0;

pub struct Tctlsf {

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

    core_mem_pool: Arc<Mutex<Tlsf>>,

    cache_hit_count: AtomicU64,
    cache_miss_count: AtomicU64,
    
}

impl Tctlsf {
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
            core_mem_pool: Arc::new(Mutex::new(Tlsf::new(DEFAULT_CORE_TLSF_ID, cmp_fl_len, cmp_sl_len, cmp_min_block_size))),
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
    static THREAD_LOCAL_A: RefCell<ThreadCacheMemPool> = RefCell::new(ThreadCacheMemPool::new( DEFAULT_TMP_MAX_SIZE, DEFAULT_TMP_FL_LEN, DEFAULT_TMP_SL_LEN, DEFAULT_TMP_MIN_BLOCK_SIZE));
}

impl ThreadSafeMemoryManager for Tctlsf {}

impl InmuteableMemoryManager for Tctlsf {
    
    fn init_mem_pool(&self, mem_pool: *mut u8, mem_pool_size: usize) {
        let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
        core_mem_pool.init_mem_pool(mem_pool, mem_pool_size);
    }

    fn allocate(&self, size: usize) -> Option<(BlockHeaderPtr, bool)> {
        let curr_thread = thread::current();
        let timer = Instant::now();
        let thread_id = curr_thread.id();
        THREAD_LOCAL_A.with(|pool| {
            // Allocate from thread memory pool
            let mut block = pool.borrow().allocate(size);
            if let Some((b, _)) = block {
                self.cache_hit_add(1);
                unsafe {
                    info!("Tctlsf[{:?}]: Allocate from cache, allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", thread_id, size, b.as_ref().size(), timer.elapsed().as_nanos());
                }
                return block;
            }
            // Allocate from core memory pool
            let mut core_mem_pool = self.core_mem_pool.lock().unwrap();
            block = core_mem_pool.allocate(size);
            if block.is_none() {
                return None;
            }

            // Let thread memory pool hold this block
            pool.borrow().hold_block(&mut block.unwrap().0);
            
            unsafe {
                info!("Tctlsf[{:?}]: Allocate from core, allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", thread_id, size, block.unwrap().0.as_ref().size(), timer.elapsed().as_nanos());
            }
            block
        })
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        let curr_thread = thread::current();
        let thread_id = curr_thread.id();

        let blcok = THREAD_LOCAL_A.with(|pool| {
            let mut new_block = pool.borrow().deallocate(block);

            // After deallocation, we should check if we should take back some blocks from thread memory pool to core memory pool
            if pool.borrow().dehold_check() {
                // Dehold it from thread memory pool
                pool.borrow().dehold_block(&mut new_block);
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
    id: u64,
    block_set: RefCell<HashSet<BlockHeaderPtr>>,
    
    /* Free blocks' total size and total num record */ 
    free_blocks_size: RefCell<usize>,
    free_blocks_num: RefCell<usize>,

     /* Holded blocks' total size and total num record */
    holded_blocks_size: RefCell<usize>,
    holded_blocks_num: RefCell<usize>,
}

pub struct ThreadCacheMemPool {
    id: u64,

    /* Tlsf parameter */
    max_size: usize,
    fl_len: usize,
    sl_len: usize,
    min_block_size: usize,

    tlsf: RefCell<Tlsf>,

    /* Free blocks' total size and total num record */ 
    free_blocks_size: RefCell<usize>,
    free_blocks_num: RefCell<usize>,

     /* Holded blocks' total size and total num record */
    holded_blocks_size: RefCell<usize>,
    holded_blocks_num: RefCell<usize>,
}

impl ThreadCacheMemPool {
    pub fn new(max_size: usize, fl_len: usize, sl_len: usize, min_block_size: usize) -> Self {
        let id = thread::current().id().as_u64().get();
        info!("ThreadCache[{:?}]: Create a new thread cache memory pool", id);
        let mut tlsf = Tlsf::new(id as usize, fl_len, sl_len, min_block_size);

        let pool = Self {
            id,
            max_size,
            fl_len,
            sl_len,
            min_block_size,
            tlsf: RefCell::new(tlsf),
            free_blocks_size: RefCell::new(0),
            free_blocks_num: RefCell::new(0),
            holded_blocks_size: RefCell::new(0),
            holded_blocks_num: RefCell::new(0),
        };
        pool
    }

    fn hold_block(&self, block: &mut BlockHeaderPtr) {
        // Hold it
        *self.holded_blocks_num.borrow_mut() += 1;
        unsafe {
            *self.holded_blocks_size.borrow_mut() += block.as_ref().size();
            // Mark holder id to block
            block.as_mut().set_holder_id(self.id as usize);
        }
    }

    /**
     * Determine if we should dehold some blocks from thread memory pool to core memory pool
     * (1) If the free blocks' total size is larger than thread memory pool's max size
     */
    fn dehold_check(&self) -> bool {
        *self.free_blocks_size.borrow() > self.max_size
    }

    fn dehold_block(&self, block: &mut BlockHeaderPtr) {
        // Remove it from tlsf
        self.tlsf.borrow_mut().remove_free_block(*block);
        // Dehold it
        *self.holded_blocks_num.borrow_mut() -= 1;
        unsafe {
            *self.holded_blocks_size.borrow_mut() -= block.as_ref().size();
            // Update free blocks' num and size record
            *self.free_blocks_num.borrow_mut() -= 1;
            *self.free_blocks_size.borrow_mut() -= block.as_ref().size();
            // Mark holder id as unholded, we shouldn't directly mark it as core memory pool's id because of concurrent issue
            block.as_mut().set_holder_id(DEFAULT_UNHOLDED_ID);
        }
    }
    
}


impl InmuteableMemoryManager for ThreadCacheMemPool {

    fn init_mem_pool(&self, _mem_pool: *mut u8, _mem_pool_size: usize) {
        panic!("ThreadCacheMemPool should not call init_mem_pool");
    }

    fn allocate(&self, size: usize) -> Option<(BlockHeaderPtr, bool)> {
        let timer = Instant::now();
        let mut tlsf = self.tlsf.borrow_mut();
        // TODO: Deal with splited block's index in cache
        let block = tlsf.allocate(size);
        let cost = timer.elapsed().as_nanos();
        if let Some((b, split)) = block {
            // Add holded blocks' num and free blocks' num if the block is splited
            if split {
                *self.free_blocks_num.borrow_mut() += 1;
                *self.holded_blocks_num.borrow_mut() += 1;
            } 
            unsafe {
                // Update free blocks' num and size record
                *self.free_blocks_num.borrow_mut() -= 1;
                *self.free_blocks_size.borrow_mut() -= b.as_ref().size();
                info!("ThreadCache[{:?}]: Allocate min size: {:?}B, real block's size: {:?}B, cost time: {:?}", self.id, size, b.as_ref().size(), cost);
            }
        } else {
            info!("ThreadCache[{:?}]: Allocate min size: {:?}B, real block's size: None, cost time: {:?}", self.id, size, cost);
        }
        block
    }

    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr {
        unsafe {
            let timer = Instant::now();
            let address = block.as_ptr() as usize;
            let block_size = block.as_ref().size();
            let next_block = block.as_ref().next_block();
            let prev_block = block.as_ref().prev_phys_block;
            let new_block = self.tlsf.borrow_mut().deallocate(block);
            if new_block != block {
                // The block has merged with previous block
                // Remove prev block from free blocks' num and holded blocks' num record
                *self.holded_blocks_num.borrow_mut() -= 1;
                *self.free_blocks_num.borrow_mut() -= 1;
            }

            if next_block.is_some() && next_block != new_block.as_ref().next_block() {
                // The block has merged with next block
                // Remove original next block from free blocks' num and holded blocks' num record
                *self.holded_blocks_num.borrow_mut() -= 1;
                *self.free_blocks_num.borrow_mut() -= 1;
            }

            // Free a merged block
            *self.free_blocks_num.borrow_mut() += 1;
            *self.free_blocks_size.borrow_mut() += block_size;

            let cost = timer.elapsed().as_nanos();
            info!("ThreadCache[{:?}]: Deallocate block[{:?}], new block[{:?}], cost time: {:?}", self.id, address, new_block.as_ptr() as usize, cost);

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
            let (block, _) = dsa.allocate(64).unwrap();

            // Allocate a block with size 128
            let (block2, _) = dsa.allocate(128).unwrap();

            assert_eq!(block.as_ref().next_block(), Some(block2));

            // Deallocate block 1
            dsa.deallocate(block);

            // Allocate a block with size 128
            let (block3, _) = dsa.allocate(128).unwrap();

            assert_eq!(block2.as_ref().next_block().unwrap(), block3);

            // Deallocate block 3
            dsa.deallocate(block3);

            // Allocate a block with 144
            let (block4, _) = dsa.allocate(144).unwrap();

            assert_eq!(block3.as_ref().next_block().unwrap(), block4);

            // Deallocate block 2
            dsa.deallocate(block2);

            // Allocate a block with 320
            let (block5, _) = dsa.allocate(320).unwrap();

            assert_eq!(block5.as_ref().next_block().unwrap(), block4);
            assert_eq!(block5, block);   
        }
    }
}