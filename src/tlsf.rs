use core::{fmt, ptr::NonNull};

use std::{fmt::Display, rc::Rc};

use bitmaps::Bitmap;
use rand::{thread_rng, Rng};

pub type Ptr<T> = Option<NonNull<T>>;

pub type BlockHeaderPtr = NonNull<BlockHeader>;

const DEFAULT_BIT_MAP_LEN: usize = 32;

const BLOCK_MAGIC_NUMBER: usize = 0x1955938454;


pub type MergeFunc<'a> = Box<dyn Fn(BlockHeaderPtr) -> bool + 'a>;

pub struct Tlsf<'a> {
    fl_len: usize,
    sl_len: usize,
    min_block_size: usize,
    min_block_size_log2: usize,
    max_block_size: usize,
    sli: usize,
    fl_bitmap: Bitmap<DEFAULT_BIT_MAP_LEN>,
    sl_bitmap: Vec<Bitmap<DEFAULT_BIT_MAP_LEN>>,
    free_list_headers: Vec<Vec<Ptr<FreeBlockHeader>>>,
    merge_func: Option<MergeFunc<'a>>,
    rand: usize,
}

unsafe impl <'a>Send for Tlsf<'a> {}
unsafe impl <'a>Sync for Tlsf<'a> {}

pub trait ThreadSafeMemoryManager : Send + Sync {
    fn init_mem_pool(&self, mem_pool: *mut u8, mem_pool_size: usize);
    fn allocate(&self, size: usize) -> Option<BlockHeaderPtr>;
    fn deallocate(&self, block: BlockHeaderPtr) -> BlockHeaderPtr;
}

pub trait MemoryManager {
    fn init_mem_pool(&mut self, addr: *mut u8, size: usize);
    fn allocate(&mut self, size: usize) -> Ptr<BlockHeader>;
    fn deallocate(&mut self, block: BlockHeaderPtr) -> BlockHeaderPtr;
}

/// The common header of a block.
/// The metadata represents the size of the block and two flags.
/// The first flag [F] indicates whether the block is free or not.
/// The second flag [L] indicates whether this block is the last block of the memory pool.
/// [size;L;F]
#[derive(Debug)]
pub struct BlockHeader {
    pub magic: usize,
    pub metadata: usize,
    pub prev_phys_block: Ptr<BlockHeader>
}

impl BlockHeader {
    pub fn new(size: usize, last: bool, free: bool, prev_phys_block: Ptr<BlockHeader>) -> Self {
        let mut metadata = size << 2;
        if last {
            metadata |= 0b10;
        }
        if free {
            metadata |= 0b01;
        }
        Self {
            magic: BLOCK_MAGIC_NUMBER,
            metadata,
            prev_phys_block
        }
    }
    
    pub fn size(&self) -> usize {
        self.metadata >> 2
    }

    fn is_last(&self) -> bool {
        self.metadata & 0b10 != 0
    }

    pub fn is_free(&self) -> bool {
        self.metadata & 0b01 != 0
    }

    fn get_prev_phys_block(&self) -> Ptr<BlockHeader> {
        self.prev_phys_block
    }

    fn set_prev_phys_block(&mut self, prev_phys_block: Ptr<BlockHeader>) {
        self.prev_phys_block = prev_phys_block;
    }

    fn set_last(&mut self, last: bool) {
        if last {
            self.metadata |= 0b10;
        } else {
            self.metadata &= !0b10;
        }
    }

    fn set_free(&mut self, free: bool) {
        if free {
            self.metadata |= 0b01;
        } else {
            self.metadata &= !0b01;
        }
    }

    fn set_size(&mut self, size: usize) {
        self.metadata = (self.metadata & 0b11) | (size << 2);
    }

    pub fn initialize(&mut self, size: usize, last: bool, free: bool, prev_phys_block: Ptr<BlockHeader>) {
        self.set_metadata(size, last, free);
        self.set_prev_phys_block(prev_phys_block);
        self.set_magic();
    }


    fn set_metadata(&mut self, size: usize, last: bool, free: bool) {
        self.metadata = size << 2;
        self.set_last(last);
        self.set_free(free);
    }

    fn set_magic(&mut self) {
        self.magic = BLOCK_MAGIC_NUMBER;
    }

    pub fn is_valid(&self) -> bool {
        self.magic == BLOCK_MAGIC_NUMBER
    }

    pub fn next_block(&self) -> Ptr<BlockHeader> {
        if self.is_last() {
            return None;
        }
        let size = self.size();
        return Some(unsafe {
            let next_addr = self as *const BlockHeader as *mut u8 as usize + size;
            NonNull::new(next_addr as *mut u8).unwrap().cast()
        });
    }
}

fn new_block_from_addr(addr: *mut u8, size: usize, last: bool, free: bool, prev_phys_block: Ptr<BlockHeader>) -> BlockHeaderPtr {
    let mut block: BlockHeaderPtr = NonNull::new(addr).unwrap().cast();
    unsafe {
        block.as_mut().initialize(size, last, free, prev_phys_block);
    }
    block.cast()
}

fn valid_size(size: usize) -> bool {
    size >= 64 && size <= 1 << 18
}

#[derive(Debug)]
struct FreeBlockHeader {
    pub block_header: BlockHeader,
    pub next_free_block: Ptr<FreeBlockHeader>,
    pub prev_free_block: Ptr<FreeBlockHeader>
}

#[derive(Debug)]
struct UsedBlockHeader {
    pub block_header: BlockHeader
}

impl <'a>Tlsf<'a> {
    pub fn with_merge_func(fl_len: usize, sl_len: usize, min_block_size: usize,
        merge_func: Option<MergeFunc<'a>>) -> Self {
        let mut sl_bitmap = Vec::with_capacity(fl_len);
        let mut free_list_headers = Vec::with_capacity(fl_len);
        for _ in 0..fl_len {
            sl_bitmap.push(Bitmap::new());
            let mut sl = Vec::with_capacity(sl_len);
            for _ in 0..sl_len {
                sl.push(None);
            }
            free_list_headers.push(sl);
        }
        let min_block_size_log2 = min_block_size.trailing_zeros() as usize;
        let max_block_size = 1 << (fl_len + min_block_size_log2);
        let sli = sl_len.trailing_zeros() as usize;
        let rand : usize = thread_rng().gen_range(0..123214798) as usize;
        let a = Self {
            fl_len,
            sl_len,
            min_block_size,
            min_block_size_log2,
            max_block_size,
            sli,
            fl_bitmap: Bitmap::new(),
            sl_bitmap,
            free_list_headers,
            merge_func,
            rand
        };
        return a;
    }

    pub fn new(fl_len: usize, sl_len: usize, min_block_size: usize) -> Self {
        Self::with_merge_func(fl_len, sl_len, min_block_size, Some(Box::new(Self::default_merge_permit)))
    }

    pub fn register_merge_func(&mut self, merge_func: MergeFunc<'a>) {
        self.merge_func = Some(merge_func);
    }

    fn merge_permit(&self, block: BlockHeaderPtr) -> bool {
        if let Some(func) = &self.merge_func {
            let aa = func;
            return aa(block);
        }
        return Self::default_merge_permit(block);
    }

    fn default_merge_permit(block: BlockHeaderPtr) -> bool {
        unsafe {
            block.as_ref().is_valid() && block.as_ref().is_free()
        }
    }

    pub fn max_block_size(&self) -> usize {
        self.max_block_size
    }

    fn map_search(&self, size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let fl_phy_index = (usize::BITS - size.leading_zeros()) as usize - 1;
        let fl = fl_phy_index - self.min_block_size_log2;

        if fl >= self.fl_len {
            return None;
        }

        let shifted_size = size >> (fl_phy_index - self.sli);
        let mask = (1 << self.sli) - 1;
        let sl = mask & shifted_size;
        Some((fl, sl))
    }

    fn map_search_bigger(&self, size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let fl_phy_index = (usize::BITS - size.leading_zeros()) as usize - 1;
        let fl = fl_phy_index - self.min_block_size_log2;

        if fl >= self.fl_len {
            return None;
        }

        let shifted_size = size >> (fl_phy_index - self.sli);
        let mask = (1 << self.sli) - 1;
        let sl = mask & shifted_size;
        if sl == self.sl_len - 1 {
            if fl == self.fl_len - 1 {
                return None;
            }
            return Some((fl + 1, 0));
        }
        Some((fl, sl + 1))
    }

    fn search_free_block(&self, min_size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let (fl, sl) = self.map_search_bigger(min_size)?;
        if self.sl_bitmap[fl].get(sl) {
            return Some((fl, sl));
        }
        if let Some(new_sl) = self.sl_bitmap[fl].next_index(sl) {
            if new_sl < self.sl_len {
                return Some((fl, new_sl));
            }
        }
        if let Some(new_fl) = self.fl_bitmap.next_index(fl) {
            let new_sl = self.sl_bitmap[new_fl].first_index()?;
            return Some((new_fl, new_sl));
        }
        None
    }

    // MIN_BLOCK_SIZE > sizeof(FreeBlockHeader)

    fn insert_free_block(&mut self, block_header: BlockHeaderPtr) {
        unsafe {
            let size = block_header.as_ref().size();
            
            debug_assert!(valid_size(size));

            let (fl, sl) = self.map_search(size).unwrap();
            let mut free_block = block_header.cast::<FreeBlockHeader>();
            debug_assert_eq!(free_block.as_ref().block_header.size(), size);
            if let Some(mut first_free_block) = self.free_list_headers[fl][sl] {
                debug_assert!(first_free_block.as_ref().block_header.is_valid());
                first_free_block.as_mut().prev_free_block = Some(free_block);
                free_block.as_mut().next_free_block = Some(first_free_block);
                free_block.as_mut().prev_free_block = None;
            } else {
                free_block.as_mut().next_free_block = None;
                free_block.as_mut().prev_free_block = None;
            }

            // set this block as the first block of the free list
            self.free_list_headers[fl][sl] = Some(free_block);

            self.sl_bitmap[fl].set(sl, true);
            self.fl_bitmap.set(fl, true);
        }
    }

    fn remove_free_block(&mut self, block_header: BlockHeaderPtr) {
        unsafe {
            let size = block_header.as_ref().size();
            let (fl, sl) = self.map_search(size).unwrap();
            let mut free_block: NonNull<FreeBlockHeader> = block_header.cast::<FreeBlockHeader>();
            debug_assert_eq!(free_block.as_ref().block_header.size(), size);
            debug_assert!(free_block.as_ref().block_header.is_valid());

            let mut new_first_block: Ptr<FreeBlockHeader> = None;
            // remove from double-link free list
            if let Some(mut next_free_block) = free_block.as_ref().next_free_block {
                debug_assert!(next_free_block.as_ref().block_header.is_valid());
                next_free_block.as_mut().prev_free_block = free_block.as_ref().prev_free_block;
            }
            if let Some(mut prev_free_block) = free_block.as_ref().prev_free_block {
                debug_assert!(prev_free_block.as_ref().block_header.is_valid());
                prev_free_block.as_mut().next_free_block = free_block.as_ref().next_free_block;
                if let Some(aa) = prev_free_block.as_ref().next_free_block {
                    let sz = aa.as_ref().block_header.size();
                    if sz == 0 || sz > 1<<18 {
                        panic!("size: {}", sz);
                    }
                    
                }
            }
            
            // update free list header in bitmap and update bitmap value
            if self.free_list_headers[fl][sl] == Some(free_block) {
                let next_free_block = free_block.as_ref().next_free_block;
                self.free_list_headers[fl][sl] = next_free_block;
                
                debug_assert!(next_free_block.is_none() || next_free_block.unwrap().as_ref().block_header.is_valid());

                if self.free_list_headers[fl][sl].is_none() {
                    self.sl_bitmap[fl].set(sl, false);
                    if self.sl_bitmap[fl].is_empty() {
                        self.fl_bitmap.set(fl, false);
                    }
                }
            }
            
            free_block.as_mut().next_free_block = None;
            free_block.as_mut().prev_free_block = None;
        }
    }

}

impl <'a>MemoryManager for Tlsf<'a> {
    fn init_mem_pool(&mut self, addr: *mut u8, size: usize) {
        let block = new_block_from_addr(addr, size, true, true, None);
        self.insert_free_block(block);
    }

    fn allocate(&mut self, min_size: usize) -> Ptr<BlockHeader> {
        let mut size = min_size + core::mem::size_of::<UsedBlockHeader>();
        if size < self.min_block_size {
            size = self.min_block_size;
        }
        
        let (fl, sl) = self.search_free_block(size)?;

        let block = self.free_list_headers[fl][sl]?;
        unsafe {
            let block_size = block.as_ref().block_header.size();
            debug_assert!(block_size >= size);

            self.remove_free_block(block.cast());

            let mut last = block.as_ref().block_header.is_last();
            let mut need_block_size = block_size;

            let mut split = false;
            let mut split_block = None;

            if block_size >= size + self.min_block_size {

                let next_phys_block = block.as_ref().block_header.next_block();

                // split the block to two blocks, one is used and the other is free
                let new_block_addr = block.cast::<u8>().as_ptr().add(size);
                let new_block = new_block_from_addr(new_block_addr, block_size - size, last, true, Some(block.cast()));
                split_block = Some(new_block);
                last = false;
                // insert new block to the free list
                debug_assert!(valid_size(new_block.as_ref().size()));
                self.insert_free_block(new_block);
                need_block_size = size;
                split = true;
                // update next block's prev_phys_block to new splitted block
                if let Some(mut next_phys_block) = next_phys_block {
                    next_phys_block.as_mut().set_prev_phys_block(Some(new_block));
                }
            }
            let mut need_block = block.cast::<UsedBlockHeader>();
            debug_assert!(need_block_size > 0);
            
            need_block.as_mut().block_header.set_metadata(need_block_size, last, false);
            
            debug_assert!(valid_size(need_block.as_ref().block_header.size()));
            debug_assert!(need_block.as_ref().block_header.is_valid());

            if let Some(next_phys_block) = need_block.as_ref().block_header.next_block() {
                debug_assert!(next_phys_block.as_ref().is_valid());
            }

            if split {
                debug_assert_eq!(need_block.as_ref().block_header.next_block(), split_block);
                debug_assert_eq!(split_block.unwrap().as_ref().prev_phys_block, Some(need_block.cast()));
            }

            // return user need block's start address
            return Some(need_block.cast());
        }
    }

    fn deallocate(&mut self, block: BlockHeaderPtr) -> BlockHeaderPtr /* after-merge block */ {
        unsafe {
            let mut new_block = block;
            let mut new_size = block.as_ref().size();
            let mut new_prev_phys_block = block.as_ref().get_prev_phys_block();
            let mut new_last = block.as_ref().is_last();


            if !block.as_ref().is_valid() {
                panic!("block is not free");
            }

            debug_assert!(block.as_ref().is_valid());

            let mut need_update_next_block = None;

            // Check if next block is free or not, if free, merge them and update the next block's next block's prev_phys_block
            if let Some(next_phys_block)  = new_block.as_ref().next_block() {

                debug_assert!(next_phys_block.as_ref().is_valid());

                if self.merge_permit(next_phys_block) {
                    need_update_next_block = next_phys_block.as_ref().next_block();

                    // remove next free block from list
                    self.remove_free_block(next_phys_block);
                    // merge the next free block and this free block
                    new_size += next_phys_block.as_ref().size();
                    new_last = next_phys_block.as_ref().is_last();
                } else {
                    need_update_next_block = Some(next_phys_block);
                }
            }

            // Check if previous block is free or not, if free, merge them
            if let Some(prev_phys_block) = block.as_ref().get_prev_phys_block() {

                debug_assert!(prev_phys_block.as_ref().is_valid());

                if self.merge_permit(prev_phys_block) {
                    new_size += prev_phys_block.as_ref().size();
                    new_prev_phys_block = prev_phys_block.as_ref().get_prev_phys_block();
                    new_block = prev_phys_block;
                    // remove previous free block from list
                    self.remove_free_block(prev_phys_block);
                }
            }

            // Create new free block
            new_block.as_mut().initialize(new_size, new_last, true, new_prev_phys_block);

            // Update next block's prev_phys_block
            if let Some(mut next_phys_block) = need_update_next_block {
                next_phys_block.as_mut().set_prev_phys_block(Some(new_block));
            }
            
            debug_assert!(new_block.as_ref().is_valid());
            debug_assert!(valid_size(new_block.as_ref().size()));

            // insert deallocated block to free list
            self.insert_free_block(new_block);
            new_block
        }
    }
}

mod tests {

    use std::mem::MaybeUninit;
    use core::mem::size_of;

    use super::*;

    #[test]
    fn test_block_header() {
        let block_header = BlockHeader::new(16, true, true, None);
        assert_eq!(block_header.size(), 16);
        assert_eq!(block_header.is_last(), true);
        assert_eq!(block_header.is_free(), true);
        assert_eq!(block_header.get_prev_phys_block(), None);

        assert_eq!(size_of::<BlockHeader>(), 24);
        assert_eq!(size_of::<FreeBlockHeader>(), 40);
        assert_eq!(size_of::<UsedBlockHeader>(), 24);
        println!("{:?}", block_header);
    }


    #[test]
    fn test_tlsf_new() {
        let tlsf = Tlsf::new(28, 4, 16);
        // println!("{:?}", tlsf);
    }

    #[test]
    fn test_tlsf_map_search() {
        let tlsf = Tlsf::new(28, 4, 16);
        assert_eq!(tlsf.map_search(16), Some((0, 0)));
        assert_eq!(tlsf.map_search(17), Some((0, 0)));
        assert_eq!(tlsf.map_search(20), Some((0, 1)));
        assert_eq!(tlsf.map_search(31), Some((0, 3)));
        assert_eq!(tlsf.map_search(32), Some((1, 0)));
        assert_eq!(tlsf.map_search(40), Some((1, 1)));
        assert_eq!(tlsf.map_search((1 << 32) - 1), Some((27, 3)));
        assert_eq!(tlsf.map_search(1 << 32), None);
    }

    #[test]
    fn test_tlsf_map_search_bigger() {
        let tlsf = Tlsf::new(28, 4, 16);
        assert_eq!(tlsf.map_search_bigger(16), Some((0, 1)));
        assert_eq!(tlsf.map_search_bigger(17), Some((0, 1)));
        assert_eq!(tlsf.map_search_bigger(20), Some((0, 2)));
        assert_eq!(tlsf.map_search_bigger(31), Some((1, 0)));
        assert_eq!(tlsf.map_search_bigger(32), Some((1, 1)));
        assert_eq!(tlsf.map_search_bigger(40), Some((1, 2)));
        assert_eq!(tlsf.map_search_bigger((1 << 32) - 1), None);
    }

    #[test]
    fn test_tlsf_search_free_block() {
        let mut tlsf = Tlsf::new(28, 4, 16);

        tlsf.fl_bitmap.set(2, true);
        tlsf.sl_bitmap[2].set(1, true);
        tlsf.sl_bitmap[2].set(3, true);

        tlsf.fl_bitmap.set(4, true);
        tlsf.sl_bitmap[4].set(0, true);
        tlsf.sl_bitmap[4].set(1, true);

        assert_eq!(tlsf.search_free_block(16), Some((2, 1)));
        assert_eq!(tlsf.search_free_block(80), Some((2, 3)));
        assert_eq!(tlsf.search_free_block(112), Some((4, 0)));
        assert_eq!(tlsf.search_free_block(128), Some((4, 0)));
        assert_eq!(tlsf.search_free_block(255), Some((4, 0)));
        assert_eq!(tlsf.search_free_block(256), Some((4, 1)));
        assert_eq!(tlsf.search_free_block(320), None);
    }

    #[test]
    fn test_cast() {
        let mut tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            let mut arena: [MaybeUninit<u8>;1024] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            block_header_ptr.as_mut().initialize(1024, true, true, None);
            assert_eq!(1024, block_header_ptr.as_ref().size());
            assert_eq!(block_header_ptr.as_ref().is_last(), true);
            assert_eq!(block_header_ptr.as_ref().is_free(), true);
            assert_eq!(block_header_ptr.as_ref().get_prev_phys_block(), None);


            let mut free_block_ptr = block_header_ptr.cast::<FreeBlockHeader>();
            assert_eq!(1024, free_block_ptr.as_ref().block_header.size());
            assert_eq!(free_block_ptr.as_ref().block_header.is_last(), true);
            assert_eq!(free_block_ptr.as_ref().block_header.is_free(), true);
            assert_eq!(free_block_ptr.as_ref().block_header.get_prev_phys_block(), None);
            assert_eq!(free_block_ptr.as_ref().next_free_block, None);
            assert_eq!(free_block_ptr.as_ref().prev_free_block, None);

            free_block_ptr.as_mut().next_free_block = Some(block_header_ptr.cast());

            let mut used_block_ptr = block_header_ptr.cast::<UsedBlockHeader>();
            used_block_ptr.as_mut().block_header.set_free(false);
            assert_eq!(1024, used_block_ptr.as_ref().block_header.size());
            assert_eq!(used_block_ptr.as_ref().block_header.is_last(), true);
            assert_eq!(used_block_ptr.as_ref().block_header.is_free(), false);
            assert_eq!(used_block_ptr.as_ref().block_header.get_prev_phys_block(), None);

            let mut free_block_ptr_2 = used_block_ptr.cast::<FreeBlockHeader>();
            assert_eq!(1024, free_block_ptr_2.as_ref().block_header.size());
            assert_eq!(free_block_ptr_2.as_ref().block_header.is_last(), true);
            assert_eq!(free_block_ptr_2.as_ref().block_header.is_free(), false);
            assert_eq!(free_block_ptr_2.as_ref().block_header.get_prev_phys_block(), None);
            assert_eq!(free_block_ptr_2.as_ref().next_free_block, Some(block_header_ptr.cast()));
            assert_eq!(free_block_ptr_2.as_ref().prev_free_block, None);

        }
    }

    #[test]
    fn test_tlsf_insert_free_block() {
        let mut tlsf: Tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            assert_eq!(tlsf.search_free_block(16), None);
            let mut arena: [MaybeUninit<u8>;1024] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            block_header_ptr.as_mut().initialize(1024, true, true, None);
            tlsf.insert_free_block(block_header_ptr);
            
            assert_eq!(tlsf.search_free_block(16), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1023), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1024), None); 
        }
    }

    #[test]
    fn test_tlsf_remove_free_block() {
        let mut tlsf: Tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            assert_eq!(tlsf.search_free_block(16), None);
            let mut arena: [MaybeUninit<u8>;1024] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            block_header_ptr.as_mut().initialize(1024, true, true, None);
            tlsf.insert_free_block(block_header_ptr);
            
            assert_eq!(tlsf.search_free_block(16), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1023), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1024), None);

            tlsf.remove_free_block(block_header_ptr);
            assert_eq!(tlsf.search_free_block(16), None);
        }
    }

    #[test]
    fn test_tlsf_insert_remove_free_block() {
        let mut tlsf: Tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            assert_eq!(tlsf.search_free_block(16), None);
            let mut arena: [MaybeUninit<u8>;10240] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr_0 : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            block_header_ptr_0.as_mut().initialize(1024, false, true, None);
            tlsf.insert_free_block(block_header_ptr_0);
            
            assert_eq!(tlsf.search_free_block(16), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1023), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1024), None);
            let mut block_header_ptr_1 : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().add(1024).cast());
            block_header_ptr_1.as_mut().initialize(1024, true, true, None);
            tlsf.insert_free_block(block_header_ptr_1);
            assert_eq!(tlsf.search_free_block(16), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1023), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1024), None);

            let mut target = tlsf.free_list_headers[6][0];

            assert_eq!(target.is_some(), true);
            assert_eq!(target.unwrap().as_ptr(), block_header_ptr_1.as_ptr().cast());
            assert_eq!(target.unwrap().as_ref().next_free_block, Some(block_header_ptr_0.cast()));
            assert_eq!(target.unwrap().as_ref().prev_free_block, None);

            tlsf.remove_free_block(block_header_ptr_1);
            assert_eq!(tlsf.search_free_block(16), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1023), Some((6, 0)));
            assert_eq!(tlsf.search_free_block(1024), None);
            
            target = tlsf.free_list_headers[6][0];
            assert_eq!(target.is_some(), true);
            assert_eq!(target.unwrap().as_ptr(), block_header_ptr_0.as_ptr().cast());
            assert_eq!(target.unwrap().as_ref().next_free_block, None);
            assert_eq!(target.unwrap().as_ref().prev_free_block, None);

            tlsf.remove_free_block(block_header_ptr_0);

            target = tlsf.free_list_headers[6][0];
            assert_eq!(target.is_none(), true);

            assert_eq!(tlsf.search_free_block(16), None);
        }
    }

    #[test]
    fn test_next_block() {
        let mut tlsf: Tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            assert_eq!(tlsf.allocate(16), None);

            let mut arena: [MaybeUninit<u8>;1024] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            tlsf.init_mem_pool(arena.as_mut_ptr().cast(), 1024);

            let ptr_0 = tlsf.allocate(128).unwrap();

            let ptr_1 = tlsf.allocate(128).unwrap();

            let ptr_2 = tlsf.allocate(128).unwrap();

            assert_eq!(ptr_0.as_ref().next_block(), Some(ptr_1));
            assert_eq!(ptr_1.as_ref().next_block(), Some(ptr_2));
        
        }
    }

    // #[test]
    fn test_allocate_deallocate() {
        let mut tlsf: Tlsf = Tlsf::new(28, 4, 16);
        unsafe {
            assert_eq!(tlsf.allocate(16), None);

            let mut arena: [MaybeUninit<u8>;1024] = MaybeUninit::uninit().assume_init();
            let mut block_header_ptr : BlockHeaderPtr = NonNull::new_unchecked(arena.as_mut_ptr().cast());
            tlsf.init_mem_pool(arena.as_mut_ptr().cast(), 1024);

            let ptr = tlsf.allocate(16);
            assert_eq!(ptr.is_some(), true);
            assert_eq!(block_header_ptr.cast(), ptr.unwrap());
            let hdr = ptr.unwrap().cast::<UsedBlockHeader>();
            assert_eq!(hdr.as_ref().block_header.size(), 40);
            assert_eq!(hdr.as_ref().block_header.is_free(), false);
            assert_eq!(hdr.as_ref().block_header.is_last(), false);
            assert_eq!(hdr.as_ref().block_header.get_prev_phys_block(), None);

            let ptr2 = tlsf.allocate(992);
            assert_eq!(ptr2.is_some(), false);

            let ptr3 = tlsf.allocate(986);
            assert_eq!(ptr3.is_some(), false);

            let ptr4 = tlsf.allocate(879);
            assert_eq!(ptr4.is_some(), true);
            let hdr4 = ptr4.unwrap().cast::<UsedBlockHeader>();
            assert_eq!(hdr4.as_ref().block_header.size(), 895);
            assert_eq!(hdr4.as_ref().block_header.is_free(), false);
            assert_eq!(hdr4.as_ref().block_header.is_last(), false);
            assert_eq!(hdr4.as_ref().block_header.get_prev_phys_block().unwrap().cast::<u8>(), hdr.cast());

            let ptr5 = tlsf.allocate(69);
            assert_eq!(ptr5.is_some(), true);
            let hdr5 = ptr5.unwrap().cast::<UsedBlockHeader>();
            assert_eq!(hdr5.as_ref().block_header.size(), 97);
            assert_eq!(hdr5.as_ref().block_header.is_free(), false);
            assert_eq!(hdr5.as_ref().block_header.is_last(), true);
            assert_eq!(hdr5.as_ref().block_header.get_prev_phys_block().unwrap().cast::<u8>(), hdr4.cast());

            assert_eq!(tlsf.allocate(16), None);

            // deallocate block[927..1024]
            tlsf.deallocate(ptr5.unwrap().cast());
            let ptr6 = tlsf.allocate(69);
            assert_eq!(ptr6.is_some(), true);
            assert_eq!(ptr5, ptr6);

            // deallocate block[0..32]
            tlsf.deallocate(ptr.unwrap().cast());
            let ptr7 = tlsf.allocate(800);
            // only have 32B free block, so can't allocate 800B
            assert_eq!(ptr7.is_none(), true);
            
            // deallocate block[32..927]
            tlsf.deallocate(ptr4.unwrap().cast());
            // now we have 895B free block and 32B free block, we expect these two blocks are merged to a 927B free block
            let ptr8 = tlsf.allocate(800);
            assert_eq!(ptr8.is_some(), true);
            let hdr8 = ptr8.unwrap().cast::<UsedBlockHeader>();
            assert_eq!(hdr8.as_ref().block_header.size(), 816);
            assert_eq!(hdr8.as_ref().block_header.is_free(), false);
            assert_eq!(block_header_ptr.cast(), ptr8.unwrap());
        }
    }
}