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
            if let Some((b, _)) = block {
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
            if let Some((b, _)) = block {
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
            if let Some((b, _)) = block {
                dsa3.deallocate(b);
            }
    }));

    group.finish();
}

criterion_group!(benches, alloc_dealloc_benchmark);
criterion_main!(benches);