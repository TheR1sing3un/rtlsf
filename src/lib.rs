mod tlsf;
mod tstlsf;
mod tctlsf;

mod tests {

    use crate::{
        tctlsf::Tctlsf, 
        tlsf::ThreadSafeMemoryManager, 
        tstlsf::Tstlsf
    };

    use std::{mem::MaybeUninit, sync::{Arc, Barrier}, time::Instant};
    use rand::{thread_rng, Rng};

    const MIN_BLOCK_SIZE: usize = 64;
    const MAX_POOL_SIZE: usize = 1 << 20;
    const FLLEN: usize = 26;
    const SLLEN: usize = 4;
    const CONCURRENT_NUM: usize = 8;
    const LOOP_TIMES: usize = 100_000;     

    #[test]
    fn bench_concurrent_allocations() {
        let dsa = Box::new(Tstlsf::new(FLLEN, SLLEN, MIN_BLOCK_SIZE));
        bench_concurrent(dsa, CONCURRENT_NUM, LOOP_TIMES);
    }

    #[test]
    fn bench_concurrent_allocations_2() {
        let dsa = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
        bench_concurrent(dsa, CONCURRENT_NUM, LOOP_TIMES);
    }


    fn bench_concurrent(dsa: Box<dyn ThreadSafeMemoryManager>, concurrent_num: usize, loop_times: usize) {
        unsafe {
            let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();

            dsa.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());

            let shared_tlsf = Arc::new(dsa);
            let mut handles = Vec::new();
            let barrier = Arc::new(Barrier::new(concurrent_num));
            for t_id in 0..concurrent_num {
                let clone_tlsf = shared_tlsf.clone();
                let clone_barrier = barrier.clone();
                let handle = std::thread::spawn(move || {
                    let mut total_time_ns = 0;
                    let mut success = 0;
                    let mut fail = 0;
                    clone_barrier.wait();

                        let pre_block = clone_tlsf.allocate(1<<8);
                        if let Some(b) = pre_block {
                            clone_tlsf.deallocate(b);
                        }

                    for _ in 0..loop_times {
                        let start = Instant::now();
                        let size : usize = thread_rng().gen_range(64..1<<8) as usize;
                        let block = clone_tlsf.allocate(size);
                        if let Some(b) = block {
                            // verify the block
                            debug_assert!(b.as_ref().is_valid());
                            debug_assert!(b.as_ref().size() >= size);
                            debug_assert!(!b.as_ref().is_free());
                            success += 1;
                            clone_tlsf.deallocate(b);
                        } else {
                            fail += 1;
                        }
                        let elapsed = start.elapsed();
                        total_time_ns += elapsed.as_nanos();
                    }
                    println!("Thread {} finished in {} ns, success: {}, fail: {}", t_id, total_time_ns, success, fail);
                    // print avg time for each allocation
                    println!("Thread {} avg time: {} ns", t_id, total_time_ns / loop_times as u128);
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