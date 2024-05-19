#![feature(thread_id_value)]

mod tlsf;
mod tstlsf;
mod tctlsf;

pub use tlsf::Tlsf;
pub use tstlsf::Tstlsf;
pub use tctlsf::Tctlsf;
pub use tlsf::MemoryManager;
pub use tlsf::ThreadSafeMemoryManager;
pub use tlsf::InmuteableMemoryManager;

#[cfg(test)]
mod tests {

    use crate::{
        tctlsf::Tctlsf, 
        tlsf::ThreadSafeMemoryManager, 
        tstlsf::Tstlsf
    };

    use std::{mem::MaybeUninit, sync::{Arc, Barrier}, time::Instant};
    use minitrace::{collector::{Config, ConsoleReporter, SpanContext}, flush, Span};
    use rand::{thread_rng, Rng};

    const MIN_BLOCK_SIZE: usize = 64;
    const MAX_POOL_SIZE: usize = 1 << 20;
    const FLLEN: usize = 26;
    const SLLEN: usize = 4;
    const CONCURRENT_NUM: usize = 8;
    const LOOP_TIMES: usize = 1000_000;

    fn preperation() {
        // env_logger::builder().filter_level(log::LevelFilter::Warn).init();
    }

    #[test]
    fn bench_single_core() {
        preperation();
        let dsa : Arc<Box<dyn ThreadSafeMemoryManager>>= Arc::new(Box::new(Tstlsf::new(FLLEN, SLLEN, MIN_BLOCK_SIZE)));
        bench_concurrent(dsa, 1, LOOP_TIMES);
    }

    #[test]
    fn bench_concurrent_allocations_1() {
        preperation();
        let dsa : Arc<Box<dyn ThreadSafeMemoryManager>>= Arc::new(Box::new(Tstlsf::new(FLLEN, SLLEN, MIN_BLOCK_SIZE)));
        bench_concurrent(dsa, CONCURRENT_NUM, LOOP_TIMES);

    }

    #[test]
    fn bench_concurrent_allocations_2() {
        preperation();
        let dsa = Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE);
        let wrap: Box<dyn ThreadSafeMemoryManager> = Box::new(dsa);
        let aa : Arc<Box<dyn ThreadSafeMemoryManager>> = Arc::new(wrap);
        bench_concurrent(aa.clone(), CONCURRENT_NUM, LOOP_TIMES);

    }


    pub fn bench_concurrent(dsa: Arc<Box<dyn ThreadSafeMemoryManager>>, concurrent_num: usize, loop_times: usize) {
        unsafe {
            let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();

            dsa.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());

            let shared_tlsf = dsa;
            let mut handles = Vec::new();
            let barrier = Arc::new(Barrier::new(concurrent_num));
            let core_ids = core_affinity::get_core_ids().unwrap();
            println!("Core ids: {:?}", core_ids);
            for t_id in 0..concurrent_num {
                let core = core_ids[t_id % core_ids.len()];
                core_affinity::set_for_current(core);
                let clone_tlsf = shared_tlsf.clone();
                let clone_barrier = barrier.clone();
                let handle = std::thread::spawn(move || {
                    let mut success = 0;
                    let mut fail = 0;
                    let start = Instant::now();
                    clone_barrier.wait();

                    for _ in 0..loop_times {
                        let size : usize = thread_rng().gen_range(64..1<<8) as usize;
                        let block = clone_tlsf.allocate(size);
                        if let Some((b, _)) = block {
                            // verify the block
                            debug_assert!(b.as_ref().is_valid());
                            debug_assert!(b.as_ref().size() >= size);
                            debug_assert!(!b.as_ref().is_free());
                            success += 1;
                            clone_tlsf.deallocate(b);
                        } else {
                            fail += 1;
                        }
                    }
                    let total_time_ns = start.elapsed().as_nanos();
                    println!("Thread {:?} finished in {} ns, success: {}, fail: {}", t_id, total_time_ns, success, fail);
                    // print avg time for each allocation
                    println!("Thread {:?} avg time: {} ns", t_id, total_time_ns / loop_times as u128);
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