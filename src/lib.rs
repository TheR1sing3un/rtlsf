mod tlsf;
// mod tctlsf;

mod tests {

    use crate::tlsf::{self, BlockHeader, MemoryManager, Ptr, Tlsf};

    use std::{cell::RefCell, mem::MaybeUninit, sync::{Arc, Barrier, Mutex}, time::Instant, usize::MIN};
    use core::ptr::NonNull;
    use rand::{thread_rng, Rng};
    
    #[test]
    fn bench_concurrent_allocations() {
        const MIN_BLOCK_SIZE: usize = 64;
        const FLLEN: usize = 26;
        const SLLEN: usize = 4;
        const CONCURRENT_NUM: usize = 4;
        const LOOP_TIMES: usize = 1_000_000; 

        let mut tlsf = Tlsf::new(FLLEN, SLLEN, MIN_BLOCK_SIZE);
        unsafe {
            let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();

            tlsf.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());

            let max_block_size = tlsf.max_block_size();

            let shared_tlsf = Arc::new(Mutex::new(tlsf));
            let mut handles = Vec::new();
            let barrier = Arc::new(Barrier::new(CONCURRENT_NUM));
            for t_id in 0..CONCURRENT_NUM {
                let clone_tlsf = shared_tlsf.clone();
                let clone_barrier = barrier.clone();
                let handle = std::thread::spawn(move || {
                    let mut total_time_ns = 0;
                    let mut success = 0;
                    let mut fail = 0;
                    clone_barrier.wait();
                    for _ in 0..LOOP_TIMES {
                        let start = Instant::now();
                        let size : usize = thread_rng().gen_range(MIN_BLOCK_SIZE..1<<18) as usize;
                        let block;
                        {
                            let mut a = clone_tlsf.lock().unwrap();
                            block = a.allocate(size);
                        }
                        if let Some(b) = block {
                            // verify the block
                            debug_assert!(b.as_ref().is_valid());
                            debug_assert!(b.as_ref().size() >= size);
                            debug_assert!(!b.as_ref().is_free());
                            success += 1;
                            let mut a = clone_tlsf.lock().unwrap();
                            a.deallocate(b);
                        } else {
                            fail += 1;
                        }
                        let elapsed = start.elapsed();
                        total_time_ns += elapsed.as_nanos();
                    }
                    println!("Thread {} finished in {} ns, success: {}, fail: {}", t_id, total_time_ns, success, fail);
                    // print avg time for each allocation
                    println!("Thread {} avg time: {} ns", t_id, total_time_ns / LOOP_TIMES as u128);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }
            println!("All threads finished");
        }
    }
}