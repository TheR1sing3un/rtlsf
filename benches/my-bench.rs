use std::sync::{Arc, Mutex};

use criterion::{criterion_group, criterion_main, Criterion};

fn lock(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock");
    let a = Arc::new(Mutex::new(0));
    group.bench_function("lock", |b| b.iter(|| {
        let mut a = a.lock().unwrap();
        *a += 1;
    }));
    group.finish();
}

criterion_group!(benches, lock);
criterion_main!(benches);