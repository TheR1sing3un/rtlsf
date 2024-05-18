use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::Barrier;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::thread_rng;
use rand::Rng;
use rtlsf::Tctlsf;
use rtlsf::{ThreadSafeMemoryManager, InmuteableMemoryManager};

const MIN_BLOCK_SIZE: usize = 64;
const MAX_POOL_SIZE: usize = 1 << 20;
const FLLEN: usize = 26;
const SLLEN: usize = 4;
const CONCURRENT_NUM: usize = 8;
const LOOP_TIMES: usize = 100_000;

pub fn concurrent(dsa: Box<dyn ThreadSafeMemoryManager>, concurrent_num: usize, loop_times: usize) {
    unsafe {
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

                    let pre_block = clone_tlsf.allocate(1<<14);
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

fn alloc_dealloc_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_dealloc_benchmark");
    
    let dsa = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
    unsafe {
        let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();
        dsa.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());
    }

    group.bench_function("small_size_alloc_dealloc", 
    |b| 
        b.iter(|| {
            let size = thread_rng().gen_range(64..256);
            let block = dsa.allocate(size);
            if let Some(b) = block {
                dsa.deallocate(b);
            }
    }));

    let dsa2 = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
    unsafe {
        let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();
        dsa2.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());
    } 

    group.bench_function("mid_size_alloc_dealloc", 
    |b| 
        b.iter(|| {
            let size = thread_rng().gen_range(256..1024);
            let block = dsa2.allocate(size);
            if let Some(b) = block {
                dsa2.deallocate(b);
            }
    }));

    let dsa3 = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
    unsafe {
        let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();
        dsa3.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());
    } 

    group.bench_function("large_size_alloc_dealloc", 
    |b| 
        b.iter(|| {
            let size = thread_rng().gen_range(1024..1<<18);
            let block = dsa3.allocate(size);
            if let Some(b) = block {
                dsa3.deallocate(b);
            }
    }));

    group.finish();
}

fn concurrent_bench(c: &mut Criterion) {

    let mut group = c.benchmark_group("concurrent_benchmark");
    
    let dsa = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
    unsafe {
        let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();
        dsa.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());
    }
    let wrap = Arc::new(dsa);

    group.bench_function("concurrent_tctlsf", 
    |b| 
        b.iter(|| {
            unsafe {
                let shared_tlsf = wrap.clone();
                let mut handles = Vec::new();
                let barrier = Arc::new(Barrier::new(4));
                for t_id in 0..4 {
                    let clone_tlsf = shared_tlsf.clone();
                    let clone_barrier = barrier.clone();
                    let handle = std::thread::spawn(move || {
                        let mut total_time_ns = 0;
                        let mut success = 0;
                        let mut fail = 0;
                        clone_barrier.wait();
        
                        for _ in 0..100 {
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
                        println!("Thread {} avg time: {} ns", t_id, total_time_ns / 100 as u128);
                    });
                    handles.push(handle);
                }
        
                for handle in handles {
                    handle.join().unwrap();
                }
                println!("All threads finished");            
            }


    }));

    let dsa2 = Box::new(Tctlsf::new(MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE, MAX_POOL_SIZE, FLLEN, SLLEN, MIN_BLOCK_SIZE));
    unsafe {
        let mut arena: [MaybeUninit<u8>;1 << 18] = MaybeUninit::uninit().assume_init();
        dsa2.init_mem_pool(arena.as_mut_ptr().cast(), arena.len());
    } 

    group.finish();

}

criterion_group!(benches, alloc_dealloc_benchmark);
criterion_main!(benches);