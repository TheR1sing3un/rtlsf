use core:: {
    ptr::NonNull,
};

use bitmaps::Bitmap;

type Ptr<T> = Option<NonNull<T>>;

const DEFAULT_BIT_MAP_LEN: usize = 32;

pub struct Tlsf<const FLLEN: usize, const SLLEN: usize, const MIN_BLOCK_SIZE: usize> {
    fl_bitmap: Bitmap<DEFAULT_BIT_MAP_LEN>,
    sl_bitmap: [Bitmap<DEFAULT_BIT_MAP_LEN>; FLLEN],
    free_list_headers: [[Ptr<FreeBlockHeader>;SLLEN];FLLEN],
}

/// The common header of a block.
/// The metadata represents the size of the block and two flags.
/// The first flag [F] indicates whether the block is free or not.
/// The second flag [L] indicates whether this block is the last block of the memory pool.
/// [size;L;F]
#[derive(Debug)]
struct BlockHeader {
    metadata: usize,
    prev_phys_block: Ptr<BlockHeader>
}

impl BlockHeader {
    fn new(size: usize, last: bool, free: bool, prev_phys_block: Ptr<BlockHeader>) -> Self {
        let mut metadata = size << 2;
        if last {
            metadata |= 0b10;
        }
        if free {
            metadata |= 0b01;
        }
        Self { metadata, prev_phys_block }
    }
    
    fn size(&self) -> usize {
        self.metadata >> 2
    }

    fn is_last(&self) -> bool {
        self.metadata & 0b10 != 0
    }

    fn is_free(&self) -> bool {
        self.metadata & 0b01 != 0
    }

    fn get_prev_phys_block(&self) -> Ptr<BlockHeader> {
        self.prev_phys_block
    }
}

#[derive(Debug)]
struct FreeBlockHeader {
    block_header: BlockHeader,
    next_free_block: Ptr<FreeBlockHeader>,
    prev_free_block: Ptr<FreeBlockHeader>
}

struct UsedBlockHeader {

    block_header: BlockHeader
}

impl<const FLLEN: usize, const SLLEN: usize, const MIN_BLOCK_SIZE: usize> 
    Tlsf<FLLEN, SLLEN, MIN_BLOCK_SIZE> {
    
    pub fn new() -> Self {
        Self {
            fl_bitmap: Bitmap::new(),
            sl_bitmap: [Bitmap::new();FLLEN],
            free_list_headers: [[None;SLLEN];FLLEN],
        }
    }

    const MIN_BLOCK_SIZE_LOG2: usize = MIN_BLOCK_SIZE.trailing_zeros() as usize;

    const MAX_BLOCK_SIZE: usize = 1 << (FLLEN + Self::MIN_BLOCK_SIZE_LOG2);

    const SLI: usize = SLLEN.trailing_zeros() as usize;

    fn map_search(&self, size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let fl_phy_index = (usize::BITS - size.leading_zeros()) as usize - 1;
        let fl = fl_phy_index - Self::MIN_BLOCK_SIZE_LOG2;

        if fl >= FLLEN {
            return None;
        }

        let shifted_size = size >> (fl_phy_index - Self::SLI);
        let mask = (1 << Self::SLI) - 1;
        let sl = mask & shifted_size;
        Some((fl, sl))
    }

    fn map_search_bigger(&self, size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let fl_phy_index = (usize::BITS - size.leading_zeros()) as usize - 1;
        let fl = fl_phy_index - Self::MIN_BLOCK_SIZE_LOG2;

        if fl >= FLLEN {
            return None;
        }

        let shifted_size = size >> (fl_phy_index - Self::SLI);
        let mask = (1 << Self::SLI) - 1;
        let sl = mask & shifted_size;
        if sl == SLLEN - 1 {
            if fl == FLLEN - 1 {
                return None;
            }
            return Some((fl + 1, 0));
        }
        Some((fl, sl + 1))
    }

    pub fn search_free_block(&self, min_size: usize) -> Option<(usize/*fl-index*/, usize/*sl-index*/)> {
        let (fl, sl) = self.map_search_bigger(min_size)?;
        if self.sl_bitmap[fl].get(sl) {
            return Some((fl, sl));
        }
        if let Some(new_sl) = self.sl_bitmap[fl].next_index(sl) {
            if new_sl < SLLEN {
                return Some((fl, new_sl));
            }
        }
        if let Some(new_fl) = self.fl_bitmap.next_index(fl) {
            let new_sl = self.sl_bitmap[new_fl].first_index()?;
            return Some((new_fl, new_sl));
        }
        None
    }

}

mod tests {

    use super::*;

    #[test]
    fn test_tlsf_new() {
        let _tlsf: Tlsf<28, 4, 16> = Tlsf::new();
    }

    #[test]
    fn test_tlsf_map_search() {
        let tlsf: Tlsf<28, 4, 16> = Tlsf::new();
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
        let tlsf: Tlsf<28, 4, 16> = Tlsf::new();
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
        let mut tlsf: Tlsf<28, 4, 16> = Tlsf::new();

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
}