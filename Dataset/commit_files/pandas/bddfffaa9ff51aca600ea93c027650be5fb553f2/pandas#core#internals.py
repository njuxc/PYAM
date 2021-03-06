import itertools
from datetime import datetime

from numpy import nan
import numpy as np

from pandas.core.index import Index, _ensure_index, _handle_legacy_indexes
import pandas.core.common as com
import pandas.lib as lib
import pandas.tslib as tslib

from pandas.util import py3compat


class Block(object):
    """
    Canonical n-dimensional unit of homogeneous dtype contained in a pandas
    data structure

    Index-ignorant; let the container take care of that
    """
    __slots__ = ['items', 'ref_items', '_ref_locs', 'values', 'ndim']
    is_numeric = False
    is_bool = False
    is_object = False
    _can_hold_na = False

    def __init__(self, values, items, ref_items, ndim=2):
        if issubclass(values.dtype.type, basestring):
            values = np.array(values, dtype=object)

        if values.ndim != ndim:
            raise AssertionError('Wrong number of dimensions')

        if len(items) != len(values):
            raise AssertionError('Wrong number of items passed (%d vs %d)'
                                 % (len(items), len(values)))

        self._ref_locs = None
        self.values = values
        self.ndim = ndim
        self.items = _ensure_index(items)
        self.ref_items = _ensure_index(ref_items)

    def _gi(self, arg):
        return self.values[arg]

    @property
    def ref_locs(self):
        if self._ref_locs is None:
            indexer = self.ref_items.get_indexer(self.items)
            indexer = com._ensure_platform_int(indexer)
            if (indexer == -1).any():
                raise AssertionError('Some block items were not in block '
                                     'ref_items')
            self._ref_locs = indexer
        return self._ref_locs

    def set_ref_items(self, ref_items, maybe_rename=True):
        """
        If maybe_rename=True, need to set the items for this guy
        """
        if not isinstance(ref_items, Index):
            raise AssertionError('block ref_items must be an Index')
        if maybe_rename:
            self.items = ref_items.take(self.ref_locs)
        self.ref_items = ref_items

    def __repr__(self):
        shape = ' x '.join([com.pprint_thing(s) for s in self.shape])
        name = type(self).__name__
        result = '%s: %s, %s, dtype %s' % (
            name, com.pprint_thing(self.items), shape, self.dtype)
        if py3compat.PY3:
            return unicode(result)
        return com.console_encode(result)

    def __contains__(self, item):
        return item in self.items

    def __len__(self):
        return len(self.values)

    def __getstate__(self):
        # should not pickle generally (want to share ref_items), but here for
        # completeness
        return (self.items, self.ref_items, self.values)

    def __setstate__(self, state):
        items, ref_items, values = state
        self.items = _ensure_index(items)
        self.ref_items = _ensure_index(ref_items)
        self.values = values
        self.ndim = values.ndim

    @property
    def shape(self):
        return self.values.shape

    @property
    def itemsize(self):
        return self.values.itemsize

    @property
    def dtype(self):
        return self.values.dtype

    def copy(self, deep=True):
        values = self.values
        if deep:
            values = values.copy()
        return make_block(values, self.items, self.ref_items)

    def merge(self, other):
        if not self.ref_items.equals(other.ref_items):
            raise AssertionError('Merge operands must have same ref_items')

        # Not sure whether to allow this or not
        # if not union_ref.equals(other.ref_items):
        #     union_ref = self.ref_items + other.ref_items
        return _merge_blocks([self, other], self.ref_items)

    def reindex_axis(self, indexer, mask, needs_masking, axis=0,
                     fill_value=np.nan):
        """
        Reindex using pre-computed indexer information
        """
        new_values = com.take_fast(self.values, indexer,
                                   mask, needs_masking, axis=axis,
                                   fill_value=fill_value)
        return make_block(new_values, self.items, self.ref_items)

    def reindex_items_from(self, new_ref_items, copy=True):
        """
        Reindex to only those items contained in the input set of items

        E.g. if you have ['a', 'b'], and the input items is ['b', 'c', 'd'],
        then the resulting items will be ['b']

        Returns
        -------
        reindexed : Block
        """
        new_ref_items, indexer = self.items.reindex(new_ref_items)
        if indexer is None:
            new_items = new_ref_items
            new_values = self.values.copy() if copy else self.values
        else:
            mask = indexer != -1
            masked_idx = indexer[mask]

            new_values = com.take_fast(self.values, masked_idx,
                                       mask=None, needs_masking=False,
                                       axis=0)
            new_items = self.items.take(masked_idx)
        return make_block(new_values, new_items, new_ref_items)

    def get(self, item):
        loc = self.items.get_loc(item)
        return self.values[loc]

    def set(self, item, value):
        """
        Modify Block in-place with new item value

        Returns
        -------
        None
        """
        loc = self.items.get_loc(item)
        self.values[loc] = value

    def delete(self, item):
        """
        Returns
        -------
        y : Block (new object)
        """
        loc = self.items.get_loc(item)
        new_items = self.items.delete(loc)
        new_values = np.delete(self.values, loc, 0)
        return make_block(new_values, new_items, self.ref_items)

    def split_block_at(self, item):
        """
        Split block into zero or more blocks around columns with given label,
        for "deleting" a column without having to copy data by returning views
        on the original array.

        Returns
        -------
        generator of Block
        """
        loc = self.items.get_loc(item)

        if type(loc) == slice or type(loc) == int:
            mask = [True] * len(self)
            mask[loc] = False
        else:  # already a mask, inverted
            mask = -loc

        for s, e in com.split_ranges(mask):
            yield make_block(self.values[s:e],
                             self.items[s:e].copy(),
                             self.ref_items)

    def fillna(self, value, inplace=False):
        if not self._can_hold_na:
            if inplace:
                return self
            else:
                return self.copy()

        new_values = self.values if inplace else self.values.copy()
        mask = com.isnull(new_values)
        np.putmask(new_values, mask, value)

        if inplace:
            return self
        else:
            return make_block(new_values, self.items, self.ref_items)

    def astype(self, dtype, copy = True, raise_on_error = True):
        """ coerce to the new type (if copy=True, return a new copy) raise on an except if raise == True """
        try:
            newb = make_block(com._astype_nansafe(self.values, dtype, copy = copy),
                              self.items, self.ref_items)
        except:
            if raise_on_error is True:
                raise
            newb = self.copy() if copy else self

        if newb.is_numeric and self.is_numeric:
            if newb.shape != self.shape or (not copy and newb.itemsize < self.itemsize):
                raise TypeError("cannot set astype for copy = [%s] for dtype (%s [%s]) with smaller itemsize that current (%s [%s])" % (copy,
                                                                                                                                        self.dtype.name,
                                                                                                                                        self.itemsize,
                                                                                                                                        newb.dtype.name,
                                                                                                                                        newb.itemsize))
        return newb

    def convert(self, copy = True, **kwargs):
        """ attempt to coerce any object types to better types
            return a copy of the block (if copy = True)
            by definition we are not an ObjectBlock here!  """

        return self.copy() if copy else self

    def _can_hold_element(self, value):
        raise NotImplementedError()

    def _try_cast(self, value):
        raise NotImplementedError()

    def _try_cast_result(self, result):
        """ try to cast the result to our original type,
        we may have roundtripped thru object in the mean-time """
        return result

    def replace(self, to_replace, value, inplace=False):
        new_values = self.values if inplace else self.values.copy()
        if self._can_hold_element(value):
            value = self._try_cast(value)

        if not isinstance(to_replace, (list, np.ndarray)):
            if self._can_hold_element(to_replace):
                to_replace = self._try_cast(to_replace)
                msk = com.mask_missing(new_values, to_replace)
                np.putmask(new_values, msk, value)
        else:
            try:
                to_replace = np.array(to_replace, dtype=self.dtype)
                msk = com.mask_missing(new_values, to_replace)
                np.putmask(new_values, msk, value)
            except Exception:
                to_replace = np.array(to_replace, dtype=object)
                for r in to_replace:
                    if self._can_hold_element(r):
                        r = self._try_cast(r)
                msk = com.mask_missing(new_values, to_replace)
                np.putmask(new_values, msk, value)

        if inplace:
            return self
        else:
            return make_block(new_values, self.items, self.ref_items)

    def putmask(self, mask, new, inplace=False):
        """ putmask the data to the block; it is possible that we may create a new dtype of block
            return the resulting block(s) """

        new_values = self.values if inplace else self.values.copy()

        # may need to align the new
        if hasattr(new, 'reindex_axis'):
            axis = getattr(new, '_het_axis', 0)
            new = new.reindex_axis(self.items, axis=axis, copy=False).values.T

        # may need to align the mask
        if hasattr(mask, 'reindex_axis'):
            axis = getattr(mask, '_het_axis', 0)
            mask = mask.reindex_axis(self.items, axis=axis, copy=False).values.T

        if self._can_hold_element(new):
            new = self._try_cast(new)
            np.putmask(new_values, mask, new)
        # upcast me
        else:
            # type of the new block
            if ((isinstance(new, np.ndarray) and issubclass(new.dtype, np.number)) or
                    isinstance(new, float)):
                typ = float
            else:
                typ = object

            # we need to exiplicty astype here to make a copy
            new_values = new_values.astype(typ)

            # we create a new block type
            np.putmask(new_values, mask, new)
            return [ make_block(new_values, self.items, self.ref_items) ]

        if inplace:
            return [ self ]

        return [ make_block(new_values, self.items, self.ref_items) ]

    def interpolate(self, method='pad', axis=0, inplace=False,
                    limit=None, missing=None, coerce=False):

        # if we are coercing, then don't force the conversion 
        # if the block can't hold the type
        if coerce:
            if not self._can_hold_na:
                if inplace:
                    return self
                else:
                    return self.copy()
        
        values = self.values if inplace else self.values.copy()

        if values.ndim != 2:
            raise NotImplementedError

        transf = (lambda x: x) if axis == 0 else (lambda x: x.T)

        if missing is None:
            mask = None
        else:  # todo create faster fill func without masking
            mask = _mask_missing(transf(values), missing)

        if method == 'pad':
            com.pad_2d(transf(values), limit=limit, mask=mask)
        else:
            com.backfill_2d(transf(values), limit=limit, mask=mask)

        return make_block(values, self.items, self.ref_items)

    def take(self, indexer, axis=1, fill_value=np.nan):
        if axis < 1:
            raise AssertionError('axis must be at least 1, got %d' % axis)
        new_values = com.take_fast(self.values, indexer, None, False,
                                   axis=axis, fill_value=fill_value)
        return make_block(new_values, self.items, self.ref_items)

    def get_values(self, dtype):
        return self.values

    def diff(self, n):
        """ return block for the diff of the values """
        new_values = com.diff(self.values, n, axis=1)
        return make_block(new_values, self.items, self.ref_items)

    def shift(self, indexer, periods):
        """ shift the block by periods, possibly upcast """

        new_values = self.values.take(indexer, axis=1)
        # convert integer to float if necessary. need to do a lot more than
        # that, handle boolean etc also
        new_values = com._maybe_upcast(new_values)
        if periods > 0:
            new_values[:, :periods] = np.nan
        else:
            new_values[:, periods:] = np.nan
        return make_block(new_values, self.items, self.ref_items)

    def eval(self, func, other, raise_on_error = True, try_cast = False):
        """ 
        evaluate the block; return result block from the result 

        Parameters
        ----------
        func  : how to combine self, other
        other : a ndarray/object
        raise_on_error : if True, raise when I can't perform the function, False by default (and just return
             the data that we had coming in)

        Returns
        -------
        a new block, the result of the func
        """
        values = self.values

        # see if we can align other
        if hasattr(other, 'reindex_axis'):
            axis = getattr(other, '_het_axis', 0)
            other = other.reindex_axis(self.items, axis=axis, copy=True).values

        # make sure that we can broadcast
        is_transposed = False
        if hasattr(other, 'ndim') and hasattr(values, 'ndim'):
            if values.ndim != other.ndim or values.shape == other.shape[::-1]:
                values = values.T
                is_transposed = True

        args = [ values, other ]
        try:
            result = func(*args)
        except:
            if raise_on_error:
                raise TypeError('Coulnd not operate %s with block values'
                                % repr(other))
            else:
                # return the values
                result = np.empty(values.shape,dtype='O')
                result.fill(np.nan)

        if not isinstance(result, np.ndarray):
            raise TypeError('Could not compare %s with block values'
                            % repr(other))

        if is_transposed:
            result = result.T

        # try to cast if requested
        if try_cast:
            result = self._try_cast_result(result)

        return make_block(result, self.items, self.ref_items)

    def where(self, other, cond, raise_on_error = True, try_cast = False):
        """ 
        evaluate the block; return result block(s) from the result 

        Parameters
        ----------
        other : a ndarray/object
        cond  : the condition to respect
        raise_on_error : if True, raise when I can't perform the function, False by default (and just return
             the data that we had coming in)

        Returns
        -------
        a new block(s), the result of the func
        """

        values = self.values

        # see if we can align other
        if hasattr(other,'reindex_axis'):
            axis = getattr(other,'_het_axis',0)
            other = other.reindex_axis(self.items, axis=axis, copy=True).values

        # make sure that we can broadcast
        is_transposed = False
        if hasattr(other, 'ndim') and hasattr(values, 'ndim'):
            if values.ndim != other.ndim or values.shape == other.shape[::-1]:
                values = values.T
                is_transposed = True

        # see if we can align cond
        if not hasattr(cond,'shape'):
            raise ValueError("where must have a condition that is ndarray like")
        if hasattr(cond,'reindex_axis'):
            axis = getattr(cond,'_het_axis',0)
            cond = cond.reindex_axis(self.items, axis=axis, copy=True).values
        else:
            cond = cond.values

        # may need to undo transpose of values
        if hasattr(values, 'ndim'):
            if values.ndim != cond.ndim or values.shape == cond.shape[::-1]:
                values = values.T
                is_transposed =  not is_transposed

        # our where function
        def func(c,v,o):
            if c.flatten().all():
                return v
            
            try:
                return np.where(c,v,o)
            except:
                if raise_on_error:
                    raise TypeError('Coulnd not operate %s with block values'
                                    % repr(o))
                else:
                    # return the values
                    result = np.empty(v.shape,dtype='O')
                    result.fill(np.nan)
                    return result

        def create_block(result, items, transpose = True):
            if not isinstance(result, np.ndarray):
                raise TypeError('Could not compare %s with block values'
                                % repr(other))

            if transpose and is_transposed:
                result = result.T

            # try to cast if requested
            if try_cast:
                result = self._try_cast_result(result)

            return make_block(result, items, self.ref_items)

        # see if we can operate on the entire block, or need item-by-item
        if cond.all().any():
            result_blocks = []
            for item in self.items:
                loc  = self.items.get_loc(item)
                item = self.items.take([loc])
                v    = values.take([loc])
                c    = cond.take([loc])
                o    = other.take([loc]) if hasattr(other,'shape') else other

                result = func(c,v,o)
                if len(result) == 1:
                    result = np.repeat(result,self.shape[1:])

                result = result.reshape(((1,) + self.shape[1:]))
                result_blocks.append(create_block(result, item, transpose = False))

            return result_blocks
        else:
            result = func(cond,values,other)
            return create_block(result, self.items)

def _mask_missing(array, missing_values):
    if not isinstance(missing_values, (list, np.ndarray)):
        missing_values = [missing_values]

    mask = None
    missing_values = np.array(missing_values, dtype=object)
    if com.isnull(missing_values).any():
        mask = com.isnull(array)
        missing_values = missing_values[com.notnull(missing_values)]

    for v in missing_values:
        if mask is None:
            mask = array == missing_values
        else:
            mask |= array == missing_values
    return mask

class NumericBlock(Block):
    is_numeric = True
    _can_hold_na = True

class FloatBlock(NumericBlock):

    def _can_hold_element(self, element):
        if isinstance(element, np.ndarray):
            return issubclass(element.dtype.type, (np.floating, np.integer))
        return isinstance(element, (float, int))

    def _try_cast(self, element):
        try:
            return float(element)
        except:  # pragma: no cover
            return element

    def should_store(self, value):
        # when inserting a column should not coerce integers to floats
        # unnecessarily
        return issubclass(value.dtype.type, np.floating) and value.dtype == self.dtype


class ComplexBlock(NumericBlock):

    def _can_hold_element(self, element):
        return isinstance(element, complex)

    def _try_cast(self, element):
        try:
            return complex(element)
        except:  # pragma: no cover
            return element

    def should_store(self, value):
        return issubclass(value.dtype.type, np.complexfloating)


class IntBlock(NumericBlock):
    _can_hold_na = False

    def _can_hold_element(self, element):
        if isinstance(element, np.ndarray):
            return issubclass(element.dtype.type, np.integer)
        return com.is_integer(element)

    def _try_cast(self, element):
        try:
            return int(element)
        except:  # pragma: no cover
            return element

    def _try_cast_result(self, result):
        # this is quite restrictive to convert
        try:
            if (isinstance(result, np.ndarray) and
                    issubclass(result.dtype.type, np.floating)):
                if com.notnull(result).all():
                    new_result = result.astype(self.dtype)
                    if (new_result == result).all():
                        return new_result
        except:
            pass

        return result

    def should_store(self, value):
        return com.is_integer_dtype(value) and value.dtype == self.dtype


class BoolBlock(Block):
    is_bool = True
    _can_hold_na = False

    def _can_hold_element(self, element):
        return isinstance(element, (int, bool))

    def _try_cast(self, element):
        try:
            return bool(element)
        except:  # pragma: no cover
            return element

    def should_store(self, value):
        return issubclass(value.dtype.type, np.bool_)


class ObjectBlock(Block):
    is_object = True
    _can_hold_na = True

    @property
    def is_bool(self):
        """ we can be a bool if we have only bool values but are of type object """
        return lib.is_bool_array(self.values.flatten())

    def convert(self, convert_dates = True, convert_numeric = True, copy = True):
        """ attempt to coerce any object types to better types
            return a copy of the block (if copy = True)
            by definition we ARE an ObjectBlock!!!!!

            can return multiple blocks!
            """

        # attempt to create new type blocks
        blocks = []
        for i, c in enumerate(self.items):
            values = self.get(c)

            values = com._possibly_convert_objects(values, convert_dates=convert_dates, convert_numeric=convert_numeric)
            values = values.reshape(((1,) + values.shape))
            items = self.items.take([i])
            newb = make_block(values, items, self.ref_items)
            blocks.append(newb)

        return blocks

    def _can_hold_element(self, element):
        return True

    def _try_cast(self, element):
        return element

    def should_store(self, value):
        return not issubclass(value.dtype.type,
                              (np.integer, np.floating, np.complexfloating,
                               np.datetime64, np.bool_))

_NS_DTYPE = np.dtype('M8[ns]')


class DatetimeBlock(Block):
    _can_hold_na = True

    def __init__(self, values, items, ref_items, ndim=2):
        if values.dtype != _NS_DTYPE:
            values = tslib.cast_to_nanoseconds(values)

        Block.__init__(self, values, items, ref_items, ndim=ndim)

    def _gi(self, arg):
        return lib.Timestamp(self.values[arg])

    def _can_hold_element(self, element):
        return com.is_integer(element) or isinstance(element, datetime)

    def _try_cast(self, element):
        try:
            return int(element)
        except:
            return element

    def should_store(self, value):
        return issubclass(value.dtype.type, np.datetime64)

    def set(self, item, value):
        """
        Modify Block in-place with new item value

        Returns
        -------
        None
        """
        loc = self.items.get_loc(item)

        if value.dtype != _NS_DTYPE:
            value = tslib.cast_to_nanoseconds(value)

        self.values[loc] = value

    def get_values(self, dtype):
        if dtype == object:
            flat_i8 = self.values.ravel().view(np.int64)
            res = tslib.ints_to_pydatetime(flat_i8)
            return res.reshape(self.values.shape)
        return self.values


def make_block(values, items, ref_items):
    dtype = values.dtype
    vtype = dtype.type
    klass = None

    if issubclass(vtype, np.floating):
        klass = FloatBlock
    elif issubclass(vtype, np.complexfloating):
        klass = ComplexBlock
    elif issubclass(vtype, np.datetime64):
        klass = DatetimeBlock
    elif issubclass(vtype, np.integer):
        klass = IntBlock
    elif dtype == np.bool_:
        klass = BoolBlock

    # try to infer a datetimeblock
    if klass is None and np.prod(values.shape):
        flat = values.flatten()
        inferred_type = lib.infer_dtype(flat)
        if inferred_type == 'datetime':

            # we have an object array that has been inferred as datetime, so
            # convert it
            try:
                values = tslib.array_to_datetime(flat).reshape(values.shape)
                klass = DatetimeBlock
            except:  # it already object, so leave it
                pass

    if klass is None:
        klass = ObjectBlock

    return klass(values, items, ref_items, ndim=values.ndim)

# TODO: flexible with index=None and/or items=None


class BlockManager(object):
    """
    Core internal data structure to implement DataFrame

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Parameters
    ----------


    Notes
    -----
    This is *not* a public API class
    """
    __slots__ = ['axes', 'blocks', '_known_consolidated', '_is_consolidated']

    def __init__(self, blocks, axes, do_integrity_check=True):
        self.axes = [_ensure_index(ax) for ax in axes]
        self.blocks = blocks

        ndim = len(axes)
        for block in blocks:
            if ndim != block.values.ndim:
                raise AssertionError(('Number of Block dimensions (%d) must '
                                      'equal number of axes (%d)')
                                     % (block.values.ndim, ndim))

        if do_integrity_check:
            self._verify_integrity()

        self._consolidate_check()

    @classmethod
    def make_empty(self):
        return BlockManager([], [[], []])

    def __nonzero__(self):
        return True

    @property
    def ndim(self):
        return len(self.axes)

    def is_mixed_dtype(self):
        counts = set()
        for block in self.blocks:
            counts.add(block.dtype)
            if len(counts) > 1:
                return True
        return False

    def set_axis(self, axis, value):
        cur_axis = self.axes[axis]
        value = _ensure_index(value)

        if len(value) != len(cur_axis):
            raise Exception('Length mismatch (%d vs %d)'
                            % (len(value), len(cur_axis)))
        self.axes[axis] = value

        if axis == 0:
            for block in self.blocks:
                block.set_ref_items(self.items, maybe_rename=True)

    # make items read only for now
    def _get_items(self):
        return self.axes[0]
    items = property(fget=_get_items)

    def __getstate__(self):
        block_values = [b.values for b in self.blocks]
        block_items = [b.items for b in self.blocks]
        axes_array = [ax for ax in self.axes]
        return axes_array, block_values, block_items

    def __setstate__(self, state):
        # discard anything after 3rd, support beta pickling format for a little
        # while longer
        ax_arrays, bvalues, bitems = state[:3]

        self.axes = [_ensure_index(ax) for ax in ax_arrays]
        self.axes = _handle_legacy_indexes(self.axes)

        self._is_consolidated = False
        self._known_consolidated = False

        blocks = []
        for values, items in zip(bvalues, bitems):
            blk = make_block(values, items, self.axes[0])
            blocks.append(blk)
        self.blocks = blocks

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        output = 'BlockManager'
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += '\nItems: %s' % ax
            else:
                output += '\nAxis %d: %s' % (i, ax)

        for block in self.blocks:
            output += '\n%s' % repr(block)
        return output

    @property
    def shape(self):
        return tuple(len(ax) for ax in self.axes)

    def _verify_integrity(self):
        mgr_shape = self.shape
        for block in self.blocks:
            if block.ref_items is not self.items:
                raise AssertionError("Block ref_items must be BlockManager "
                                     "items")
            if block.values.shape[1:] != mgr_shape[1:]:
                raise AssertionError('Block shape incompatible with manager')
        tot_items = sum(len(x.items) for x in self.blocks)
        if len(self.items) != tot_items:
            raise AssertionError('Number of manager items must equal union of '
                                 'block items')

    def apply(self, f, *args, **kwargs):
        """ iterate over the blocks, collect and create a new block manager """
        axes = kwargs.pop('axes',None)
        result_blocks = []
        for blk in self.blocks:
            if callable(f):
                applied = f(blk, *args, **kwargs)
            else:
                applied = getattr(blk,f)(*args, **kwargs)

            if isinstance(applied,list):
                result_blocks.extend(applied)
            else:
                result_blocks.append(applied)
        bm = self.__class__(result_blocks, axes or self.axes)
        bm._consolidate_inplace()
        return bm

    def where(self, *args, **kwargs):
        return self.apply('where', *args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.apply('eval', *args, **kwargs)

    def putmask(self, *args, **kwargs):
        return self.apply('putmask', *args, **kwargs)

    def diff(self, *args, **kwargs):
        return self.apply('diff', *args, **kwargs)

    def interpolate(self, *args, **kwargs):
        return self.apply('interpolate', *args, **kwargs)

    def shift(self, *args, **kwargs):
        return self.apply('shift', *args, **kwargs)

    def fillna(self, *args, **kwargs):
        return self.apply('fillna', *args, **kwargs)

    def astype(self, *args, **kwargs):
        return self.apply('astype', *args, **kwargs)

    def convert(self, *args, **kwargs):
        return self.apply('convert', *args, **kwargs)

    def replace(self, *args, **kwargs):
        return self.apply('replace', *args, **kwargs)

    def replace_list(self, src_lst, dest_lst, inplace=False):
        """ do a list replace """
        if not inplace:
            self = self.copy()

        sset = set(src_lst)
        if any([k in sset for k in dest_lst]):
            masks = {}
            for s in src_lst:
                masks[s] = [b.values == s for b in self.blocks]

            for s, d in zip(src_lst, dest_lst):
                [b.putmask(masks[s][i], d, inplace=True) for i, b in
                 enumerate(self.blocks)]
        else:
            for s, d in zip(src_lst, dest_lst):
                self.replace(s, d, inplace=True)

        return self

    def is_consolidated(self):
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self):
        dtypes = [blk.dtype.type for blk in self.blocks]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def get_numeric_data(self, copy=False, type_list=None, as_blocks = False):
        """
        Parameters
        ----------
        copy : boolean, default False
            Whether to copy the blocks
        type_list : tuple of type, default None
            Numeric types by default (Float/Complex/Int but not Datetime)
        """
        if type_list is None:
            filter_blocks = lambda block: block.is_numeric
        else:
            type_list = self._get_clean_block_types(type_list)
            filter_blocks = lambda block: isinstance(block, type_list)

        maybe_copy = lambda b: b.copy() if copy else b
        num_blocks = [maybe_copy(b) for b in self.blocks if filter_blocks(b)]
        if as_blocks:
            return num_blocks

        if len(num_blocks) == 0:
            return BlockManager.make_empty()

        indexer = np.sort(np.concatenate([b.ref_locs for b in num_blocks]))
        new_items = self.items.take(indexer)

        new_blocks = []
        for b in num_blocks:
            b = b.copy(deep=False)
            b.ref_items = new_items
            new_blocks.append(b)
        new_axes = list(self.axes)
        new_axes[0] = new_items
        return BlockManager(new_blocks, new_axes, do_integrity_check=False)

    def _get_clean_block_types(self, type_list):
        if not isinstance(type_list, tuple):
            try:
                type_list = tuple(type_list)
            except TypeError:
                type_list = (type_list,)

        type_map = {int: IntBlock, float: FloatBlock,
                    complex: ComplexBlock,
                    np.datetime64: DatetimeBlock,
                    datetime: DatetimeBlock,
                    bool: BoolBlock,
                    object: ObjectBlock}

        type_list = tuple([type_map.get(t, t) for t in type_list])
        return type_list

    def get_bool_data(self, copy=False, as_blocks=False):
        return self.get_numeric_data(copy=copy, type_list=(BoolBlock,),
                                     as_blocks=as_blocks)

    def get_slice(self, slobj, axis=0):
        new_axes = list(self.axes)
        new_axes[axis] = new_axes[axis][slobj]

        if axis == 0:
            new_items = new_axes[0]
            if len(self.blocks) == 1:
                blk = self.blocks[0]
                newb = make_block(blk.values[slobj], new_items,
                                  new_items)
                new_blocks = [newb]
            else:
                return self.reindex_items(new_items)
        else:
            new_blocks = self._slice_blocks(slobj, axis)

        return BlockManager(new_blocks, new_axes, do_integrity_check=False)

    def _slice_blocks(self, slobj, axis):
        new_blocks = []

        slicer = [slice(None, None) for _ in range(self.ndim)]
        slicer[axis] = slobj
        slicer = tuple(slicer)

        for block in self.blocks:
            newb = make_block(block.values[slicer], block.items,
                              block.ref_items)
            new_blocks.append(newb)
        return new_blocks

    def get_series_dict(self):
        # For DataFrame
        return _blocks_to_series_dict(self.blocks, self.axes[1])

    def __contains__(self, item):
        return item in self.items

    @property
    def nblocks(self):
        return len(self.blocks)

    def copy(self, deep=True):
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : boolean, default True
            If False, return shallow copy (do not copy data)

        Returns
        -------
        copy : BlockManager
        """
        copy_blocks = [block.copy(deep=deep) for block in self.blocks]
        # copy_axes = [ax.copy() for ax in self.axes]
        copy_axes = list(self.axes)
        return BlockManager(copy_blocks, copy_axes, do_integrity_check=False)

    def as_matrix(self, items=None):
        if len(self.blocks) == 0:
            mat = np.empty(self.shape, dtype=float)
        elif len(self.blocks) == 1:
            blk = self.blocks[0]
            if items is None or blk.items.equals(items):
                # if not, then just call interleave per below
                mat = blk.values
            else:
                mat = self.reindex_items(items).as_matrix()
        else:
            if items is None:
                mat = self._interleave(self.items)
            else:
                mat = self.reindex_items(items).as_matrix()

        return mat

    def _interleave(self, items):
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        dtype = _interleaved_dtype(self.blocks)
        items = _ensure_index(items)

        result = np.empty(self.shape, dtype=dtype)
        itemmask = np.zeros(len(items), dtype=bool)

        # By construction, all of the item should be covered by one of the
        # blocks
        if items.is_unique:
            for block in self.blocks:
                indexer = items.get_indexer(block.items)
                if (indexer == -1).any():
                    raise AssertionError('Items must contain all block items')
                result[indexer] = block.get_values(dtype)
                itemmask[indexer] = 1
        else:
            for block in self.blocks:
                mask = items.isin(block.items)
                indexer = mask.nonzero()[0]
                if (len(indexer) != len(block.items)):
                    raise AssertionError('All items must be in block items')
                result[indexer] = block.get_values(dtype)
                itemmask[indexer] = 1

        if not itemmask.all():
            raise AssertionError('Some items were not contained in blocks')

        return result

    def xs(self, key, axis=1, copy=True):
        if axis < 1:
            raise AssertionError('Can only take xs across axis >= 1, got %d'
                                 % axis)

        loc = self.axes[axis].get_loc(key)
        slicer = [slice(None, None) for _ in range(self.ndim)]
        slicer[axis] = loc
        slicer = tuple(slicer)

        new_axes = list(self.axes)

        # could be an array indexer!
        if isinstance(loc, (slice, np.ndarray)):
            new_axes[axis] = new_axes[axis][loc]
        else:
            new_axes.pop(axis)

        new_blocks = []
        if len(self.blocks) > 1:
            if not copy:
                raise Exception('cannot get view of mixed-type or '
                                'non-consolidated DataFrame')
            for blk in self.blocks:
                newb = make_block(blk.values[slicer], blk.items, blk.ref_items)
                new_blocks.append(newb)
        elif len(self.blocks) == 1:
            vals = self.blocks[0].values[slicer]
            if copy:
                vals = vals.copy()
            new_blocks = [make_block(vals, self.items, self.items)]

        return BlockManager(new_blocks, new_axes)

    def fast_2d_xs(self, loc, copy=False):
        """

        """
        if len(self.blocks) == 1:
            result = self.blocks[0].values[:, loc]
            if copy:
                result = result.copy()
            return result

        if not copy:
            raise Exception('cannot get view of mixed-type or '
                            'non-consolidated DataFrame')

        dtype = _interleaved_dtype(self.blocks)

        items = self.items
        n = len(items)
        result = np.empty(n, dtype=dtype)
        for blk in self.blocks:
            for j, item in enumerate(blk.items):
                i = items.get_loc(item)
                result[i] = blk._gi((j, loc))

        return result

    def consolidate(self):
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            return self

        new_blocks = _consolidate(self.blocks, self.items)
        return BlockManager(new_blocks, self.axes)

    def _consolidate_inplace(self):
        self.blocks = _consolidate(self.blocks, self.items)
        self._is_consolidated = True
        self._known_consolidated = True

    def get(self, item):
        _, block = self._find_block(item)
        return block.get(item)

    def iget(self, i):
        item = self.items[i]
        if self.items.is_unique:
            return self.get(item)
        else:
            # ugh
            try:
                inds, = (self.items == item).nonzero()
            except AttributeError:  # MultiIndex
                inds, = self.items.map(lambda x: x == item).nonzero()

            _, block = self._find_block(item)

            try:
                binds, = (block.items == item).nonzero()
            except AttributeError:  # MultiIndex
                binds, = block.items.map(lambda x: x == item).nonzero()

            for j, (k, b) in enumerate(zip(inds, binds)):
                if i == k:
                    return block.values[b]

            raise Exception('Cannot have duplicate column names '
                            'split across dtypes')

    def get_scalar(self, tup):
        """
        Retrieve single item
        """
        item = tup[0]
        _, blk = self._find_block(item)

        # this could obviously be seriously sped up in cython
        item_loc = blk.items.get_loc(item),
        full_loc = item_loc + tuple(ax.get_loc(x)
                                    for ax, x in zip(self.axes[1:], tup[1:]))
        return blk.values[full_loc]

    def delete(self, item):
        i, _ = self._find_block(item)
        loc = self.items.get_loc(item)

        self._delete_from_block(i, item)
        if com._is_bool_indexer(loc):  # dupe keys may return mask
            loc = [i for i, v in enumerate(loc) if v]

        new_items = self.items.delete(loc)

        self.set_items_norename(new_items)
        self._known_consolidated = False

    def set(self, item, value):
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """
        if value.ndim == self.ndim - 1:
            value = value.reshape((1,) + value.shape)
        if value.shape[1:] != self.shape[1:]:
            raise AssertionError('Shape of new values must be compatible '
                                 'with manager shape')

        def _set_item(item, arr):
            i, block = self._find_block(item)
            if not block.should_store(value):
                # delete from block, create and append new block
                self._delete_from_block(i, item)
                self._add_new_block(item, arr, loc=None)
            else:
                block.set(item, arr)

        try:
            loc = self.items.get_loc(item)
            if isinstance(loc, int):
                _set_item(self.items[loc], value)
            else:
                subset = self.items[loc]
                if len(value) != len(subset):
                    raise AssertionError(
                        'Number of items to set did not match')
                for i, (item, arr) in enumerate(zip(subset, value)):
                    _set_item(item, arr[None, :])
        except KeyError:
            # insert at end
            self.insert(len(self.items), item, value)

        self._known_consolidated = False

    def insert(self, loc, item, value):
        if item in self.items:
            raise Exception('cannot insert %s, already exists' % item)

        new_items = self.items.insert(loc, item)
        self.set_items_norename(new_items)

        # new block
        self._add_new_block(item, value, loc=loc)

        if len(self.blocks) > 100:
            self._consolidate_inplace()

        self._known_consolidated = False

    def set_items_norename(self, value):
        value = _ensure_index(value)
        self.axes[0] = value

        for block in self.blocks:
            block.set_ref_items(value, maybe_rename=False)

    def _delete_from_block(self, i, item):
        """
        Delete and maybe remove the whole block
        """
        block = self.blocks.pop(i)
        for b in block.split_block_at(item):
            self.blocks.append(b)

    def _add_new_block(self, item, value, loc=None):
        # Do we care about dtype at the moment?

        # hm, elaborate hack?
        if loc is None:
            loc = self.items.get_loc(item)
        new_block = make_block(value, self.items[loc:loc + 1].copy(),
                               self.items)
        self.blocks.append(new_block)

    def _find_block(self, item):
        self._check_have(item)
        for i, block in enumerate(self.blocks):
            if item in block:
                return i, block

    def _check_have(self, item):
        if item not in self.items:
            raise KeyError('no item named %s' % com.pprint_thing(item))

    def reindex_axis(self, new_axis, method=None, axis=0, copy=True):
        new_axis = _ensure_index(new_axis)
        cur_axis = self.axes[axis]

        if new_axis.equals(cur_axis):
            if copy:
                result = self.copy(deep=True)
                result.axes[axis] = new_axis

                if axis == 0:
                    # patch ref_items, #1823
                    for blk in result.blocks:
                        blk.ref_items = new_axis

                return result
            else:
                return self

        if axis == 0:
            if method is not None:
                raise AssertionError('method argument not supported for '
                                     'axis == 0')
            return self.reindex_items(new_axis)

        new_axis, indexer = cur_axis.reindex(new_axis, method)
        return self.reindex_indexer(new_axis, indexer, axis=axis)

    def reindex_indexer(self, new_axis, indexer, axis=1, fill_value=np.nan):
        """
        pandas-indexer with -1's only.
        """
        if axis == 0:
            return self._reindex_indexer_items(new_axis, indexer, fill_value)

        mask = indexer == -1

        # TODO: deal with length-0 case? or does it fall out?
        needs_masking = len(new_axis) > 0 and mask.any()

        new_blocks = []
        for block in self.blocks:
            newb = block.reindex_axis(indexer, mask, needs_masking,
                                      axis=axis, fill_value=fill_value)
            new_blocks.append(newb)

        new_axes = list(self.axes)
        new_axes[axis] = new_axis
        return BlockManager(new_blocks, new_axes)

    def _reindex_indexer_items(self, new_items, indexer, fill_value):
        # TODO: less efficient than I'd like

        item_order = com.take_1d(self.items.values, indexer)

        # keep track of what items aren't found anywhere
        mask = np.zeros(len(item_order), dtype=bool)

        new_blocks = []
        for blk in self.blocks:
            blk_indexer = blk.items.get_indexer(item_order)
            selector = blk_indexer != -1
            # update with observed items
            mask |= selector

            if not selector.any():
                continue

            new_block_items = new_items.take(selector.nonzero()[0])
            new_values = com.take_fast(blk.values, blk_indexer[selector],
                                       None, False, axis=0)
            new_blocks.append(make_block(new_values, new_block_items,
                                         new_items))

        if not mask.all():
            na_items = new_items[-mask]
            na_block = self._make_na_block(na_items, new_items,
                                           fill_value=fill_value)
            new_blocks.append(na_block)
            new_blocks = _consolidate(new_blocks, new_items)

        return BlockManager(new_blocks, [new_items] + self.axes[1:])

    def reindex_items(self, new_items, copy=True, fill_value=np.nan):
        """

        """
        new_items = _ensure_index(new_items)
        data = self
        if not data.is_consolidated():
            data = data.consolidate()
            return data.reindex_items(new_items)

        # TODO: this part could be faster (!)
        new_items, indexer = self.items.reindex(new_items)

        # could have some pathological (MultiIndex) issues here
        new_blocks = []
        if indexer is None:
            for blk in self.blocks:
                if copy:
                    new_blocks.append(blk.reindex_items_from(new_items))
                else:
                    blk.ref_items = new_items
                    new_blocks.append(blk)
        else:
            for block in self.blocks:
                newb = block.reindex_items_from(new_items, copy=copy)
                if len(newb.items) > 0:
                    new_blocks.append(newb)

            mask = indexer == -1
            if mask.any():
                extra_items = new_items[mask]
                na_block = self._make_na_block(extra_items, new_items,
                                               fill_value=fill_value)
                new_blocks.append(na_block)
                new_blocks = _consolidate(new_blocks, new_items)

        return BlockManager(new_blocks, [new_items] + self.axes[1:])

    def _make_na_block(self, items, ref_items, fill_value=np.nan):
        # TODO: infer dtypes other than float64 from fill_value

        block_shape = list(self.shape)
        block_shape[0] = len(items)

        dtype = com._infer_dtype(fill_value)
        block_values = np.empty(block_shape, dtype=dtype)
        block_values.fill(fill_value)
        na_block = make_block(block_values, items, ref_items)
        return na_block

    def take(self, indexer, axis=1):
        if axis == 0:
            raise NotImplementedError

        indexer = com._ensure_platform_int(indexer)

        n = len(self.axes[axis])
        if ((indexer == -1) | (indexer >= n)).any():
            raise Exception('Indices must be nonzero and less than '
                            'the axis length')

        new_axes = list(self.axes)
        new_axes[axis] = self.axes[axis].take(indexer)
        new_blocks = []
        for blk in self.blocks:
            new_values = com.take_fast(blk.values, indexer, None, False,
                                       axis=axis)
            newb = make_block(new_values, blk.items, self.items)
            new_blocks.append(newb)

        return BlockManager(new_blocks, new_axes)

    def merge(self, other, lsuffix=None, rsuffix=None):
        if not self._is_indexed_like(other):
            raise AssertionError('Must have same axes to merge managers')

        this, other = self._maybe_rename_join(other, lsuffix, rsuffix)

        cons_items = this.items + other.items
        consolidated = _consolidate(this.blocks + other.blocks, cons_items)

        new_axes = list(this.axes)
        new_axes[0] = cons_items

        return BlockManager(consolidated, new_axes)

    def _maybe_rename_join(self, other, lsuffix, rsuffix, copydata=True):
        to_rename = self.items.intersection(other.items)
        if len(to_rename) > 0:
            if not lsuffix and not rsuffix:
                raise Exception('columns overlap: %s' % to_rename)

            def lrenamer(x):
                if x in to_rename:
                    return '%s%s' % (x, lsuffix)
                return x

            def rrenamer(x):
                if x in to_rename:
                    return '%s%s' % (x, rsuffix)
                return x

            this = self.rename_items(lrenamer, copydata=copydata)
            other = other.rename_items(rrenamer, copydata=copydata)
        else:
            this = self

        return this, other

    def _is_indexed_like(self, other):
        """
        Check all axes except items
        """
        if self.ndim != other.ndim:
            raise AssertionError(('Number of dimensions must agree '
                                  'got %d and %d') % (self.ndim, other.ndim))
        for ax, oax in zip(self.axes[1:], other.axes[1:]):
            if not ax.equals(oax):
                return False
        return True

    def rename_axis(self, mapper, axis=1):
        new_axis = Index([mapper(x) for x in self.axes[axis]])
        if not new_axis.is_unique:
            raise AssertionError('New axis must be unique to rename')

        new_axes = list(self.axes)
        new_axes[axis] = new_axis
        return BlockManager(self.blocks, new_axes)

    def rename_items(self, mapper, copydata=True):
        new_items = Index([mapper(x) for x in self.items])
        new_items.is_unique

        new_blocks = []
        for block in self.blocks:
            newb = block.copy(deep=copydata)
            newb.set_ref_items(new_items, maybe_rename=True)
            new_blocks.append(newb)
        new_axes = list(self.axes)
        new_axes[0] = new_items
        return BlockManager(new_blocks, new_axes)

    def add_prefix(self, prefix):
        f = (('%s' % prefix) + '%s').__mod__
        return self.rename_items(f)

    def add_suffix(self, suffix):
        f = ('%s' + ('%s' % suffix)).__mod__
        return self.rename_items(f)

    @property
    def block_id_vector(self):
        # TODO
        result = np.empty(len(self.items), dtype=int)
        result.fill(-1)

        for i, blk in enumerate(self.blocks):
            indexer = self.items.get_indexer(blk.items)
            if (indexer == -1).any():
                raise AssertionError('Block items must be in manager items')
            result.put(indexer, i)

        if (result < 0).any():
            raise AssertionError('Some items were not in any block')
        return result

    @property
    def item_dtypes(self):
        result = np.empty(len(self.items), dtype='O')
        mask = np.zeros(len(self.items), dtype=bool)
        for i, blk in enumerate(self.blocks):
            indexer = self.items.get_indexer(blk.items)
            result.put(indexer, blk.values.dtype.name)
            mask.put(indexer, 1)
        if not (mask.all()):
            raise AssertionError('Some items were not in any block')
        return result


def form_blocks(arrays, names, axes):
    # pre-filter out items if we passed it
    items = axes[0]

    if len(arrays) < len(items):
        extra_items = items - Index(names)
    else:
        extra_items = []

    # put "leftover" items in float bucket, where else?
    # generalize?
    float_items = []
    complex_items = []
    int_items = []
    bool_items = []
    object_items = []
    datetime_items = []
    for k, v in zip(names, arrays):
        if issubclass(v.dtype.type, np.floating):
            float_items.append((k, v))
        elif issubclass(v.dtype.type, np.complexfloating):
            complex_items.append((k, v))
        elif issubclass(v.dtype.type, np.datetime64):
            if v.dtype != _NS_DTYPE:
                v = tslib.cast_to_nanoseconds(v)

            if hasattr(v, 'tz') and v.tz is not None:
                object_items.append((k, v))
            else:
                datetime_items.append((k, v))
        elif issubclass(v.dtype.type, np.integer):
            if v.dtype == np.uint64:
                # HACK #2355 definite overflow
                if (v > 2 ** 63 - 1).any():
                    object_items.append((k, v))
                    continue
            int_items.append((k, v))
        elif v.dtype == np.bool_:
            bool_items.append((k, v))
        else:
            object_items.append((k, v))

    blocks = []
    if len(float_items):
        float_blocks = _multi_blockify(float_items, items)
        blocks.extend(float_blocks)

    if len(complex_items):
        complex_blocks = _simple_blockify(complex_items, items, np.complex128)
        blocks.extend(complex_blocks)

    if len(int_items):
        int_blocks = _multi_blockify(int_items, items)
        blocks.extend(int_blocks)

    if len(datetime_items):
        datetime_blocks = _simple_blockify(datetime_items, items, _NS_DTYPE)
        blocks.extend(datetime_blocks)

    if len(bool_items):
        bool_blocks = _simple_blockify(bool_items, items, np.bool_)
        blocks.extend(bool_blocks)

    if len(object_items) > 0:
        object_blocks = _simple_blockify(object_items, items, np.object_)
        blocks.extend(object_blocks)

    if len(extra_items):
        shape = (len(extra_items),) + tuple(len(x) for x in axes[1:])

        # empty items -> dtype object
        block_values = np.empty(shape, dtype=object)

        block_values.fill(nan)

        na_block = make_block(block_values, extra_items, items)
        blocks.append(na_block)
        blocks = _consolidate(blocks, items)

    return blocks


def _simple_blockify(tuples, ref_items, dtype):
    """ return a single array of a block that has a single dtype; if dtype is not None, coerce to this dtype """
    block_items, values = _stack_arrays(tuples, ref_items, dtype)

    # CHECK DTYPE?
    if dtype is not None and values.dtype != dtype:  # pragma: no cover
        values = values.astype(dtype)

    return [ make_block(values, block_items, ref_items) ]


def _multi_blockify(tuples, ref_items, dtype = None):
    """ return an array of blocks that potentially have different dtypes """

    # group by dtype
    grouper = itertools.groupby(tuples, lambda x: x[1].dtype)

    new_blocks = []
    for dtype, tup_block in grouper:

        block_items, values = _stack_arrays(list(tup_block), ref_items, dtype)
        block = make_block(values, block_items, ref_items)
        new_blocks.append(block)

    return new_blocks

def _stack_arrays(tuples, ref_items, dtype):
    from pandas.core.series import Series

    # fml
    def _asarray_compat(x):
        # asarray shouldn't be called on SparseSeries
        if isinstance(x, Series):
            return x.values
        else:
            return np.asarray(x)

    def _shape_compat(x):
        # sparseseries
        if isinstance(x, Series):
            return len(x),
        else:
            return x.shape

    names, arrays = zip(*tuples)

    # index may box values
    items = ref_items[ref_items.isin(names)]

    first = arrays[0]
    shape = (len(arrays),) + _shape_compat(first)

    stacked = np.empty(shape, dtype=dtype)
    for i, arr in enumerate(arrays):
        stacked[i] = _asarray_compat(arr)

    return items, stacked


def _blocks_to_series_dict(blocks, index=None):
    from pandas.core.series import Series

    series_dict = {}

    for block in blocks:
        for item, vec in zip(block.items, block.values):
            series_dict[item] = Series(vec, index=index, name=item)
    return series_dict


def _interleaved_dtype(blocks):
    if not len(blocks): return None

    from collections import defaultdict
    counts = defaultdict(lambda: [])
    for x in blocks:
        counts[type(x)].append(x)

    def _lcd_dtype(l):
        """ find the lowest dtype that can accomodate the given types """
        m = l[0].dtype
        for x in l[1:]:
            if x.dtype.itemsize > m.itemsize:
                m = x.dtype
        return m

    have_int = len(counts[IntBlock]) > 0
    have_bool = len(counts[BoolBlock]) > 0
    have_object = len(counts[ObjectBlock]) > 0
    have_float = len(counts[FloatBlock]) > 0
    have_complex = len(counts[ComplexBlock]) > 0
    have_dt64 = len(counts[DatetimeBlock]) > 0
    have_numeric = have_float or have_complex or have_int

    if (have_object or
        (have_bool and have_numeric) or
            (have_numeric and have_dt64)):
        return np.dtype(object)
    elif have_bool:
        return np.dtype(bool)
    elif have_int and not have_float and not have_complex:
        return _lcd_dtype(counts[IntBlock])
    elif have_dt64 and not have_float and not have_complex:
        return np.dtype('M8[ns]')
    elif have_complex:
        return np.dtype('c16')
    else:
        return _lcd_dtype(counts[FloatBlock])


def _consolidate(blocks, items):
    """
    Merge blocks having same dtype
    """
    get_dtype = lambda x: x.dtype.name

    # sort by dtype
    grouper = itertools.groupby(sorted(blocks, key=get_dtype),
                                lambda x: x.dtype)

    new_blocks = []
    for dtype, group_blocks in grouper:
        new_block = _merge_blocks(list(group_blocks), items)
        new_blocks.append(new_block)

    return new_blocks


def _merge_blocks(blocks, items):
    if len(blocks) == 1:
        return blocks[0]
    new_values = _vstack([b.values for b in blocks])
    new_items = blocks[0].items.append([b.items for b in blocks[1:]])
    new_block = make_block(new_values, new_items, items)
    return new_block.reindex_items_from(items)


def _vstack(to_stack):
    if all(x.dtype == _NS_DTYPE for x in to_stack):
        # work around NumPy 1.6 bug
        new_values = np.vstack([x.view('i8') for x in to_stack])
        return new_values.view(_NS_DTYPE)
    else:
        return np.vstack(to_stack)
