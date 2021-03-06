# -*- coding: utf-8 -*-
"""Utilities to evaluate the predictive performance of models

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
# License: BSD 3 clause

from __future__ import division

import warnings
import numpy as np

from scipy.sparse import coo_matrix
from scipy.spatial.distance import hamming as sp_hamming

from ..externals.six.moves import zip
from ..preprocessing import LabelBinarizer
from ..utils import check_arrays
from ..utils import deprecated
from ..utils.fixes import divide
from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target


###############################################################################
# General utilities
###############################################################################
def _is_1d(x):
    """Return True if x is 1d or a column vector

    Parameters
    ----------
    x : numpy array.

    Returns
    -------
    is_1d : boolean,
        Return True if x can be considered as a 1d vector.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics.metrics import _is_1d
    >>> _is_1d([1, 2, 3])
    True
    >>> _is_1d(np.array([1, 2, 3]))
    True
    >>> _is_1d([[1, 2, 3]])
    False
    >>> _is_1d(np.array([[1, 2, 3]]))
    False
    >>> _is_1d([[1], [2], [3]])
    True
    >>> _is_1d(np.array([[1], [2], [3]]))
    True
    >>> _is_1d([[1, 2], [3, 4]])
    False
    >>> _is_1d(np.array([[1, 2], [3, 4]]))
    False

    See also
    --------
    _check_1d_array

    """
    shape = np.shape(x)
    return len(shape) == 1 or len(shape) == 2 and shape[1] == 1


def _check_1d_array(y1, y2, ravel=False):
    """Check that y1 and y2 are vectors of the same shape.

    It convert 1d arrays (y1 and y2) of various shape to a common shape
    representation. Note that ``y1`` and ``y2`` should have the same number of
    elements.

    Parameters
    ----------
    y1 : array-like,
        y1 must be a "vector".

    y2 : array-like
        y2 must be a "vector".

    ravel : boolean, optional (default=False),
        If ``ravel``` is set to ``True``, then ``y1`` and ``y2`` are raveled.

    Returns
    -------
    y1 : numpy array,
        If ``ravel`` is set to ``True``, return np.ravel(y1), else
        return y1.

    y2 : numpy array,
        Return y2  reshaped to have the shape of y1.

    Examples
    --------
    >>> from sklearn.metrics.metrics import _check_1d_array
    >>> _check_1d_array([1, 2], [[3], [4]])
    (array([1, 2]), array([3, 4]))

    See also
    --------
    _is_1d

    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    if not _is_1d(y1):
        raise ValueError("y1 can't be considered as a vector")

    if not _is_1d(y2):
        raise ValueError("y2 can't be considered as a vector")

    if ravel:
        return np.ravel(y1), np.ravel(y2)
    else:
        if np.shape(y1) != np.shape(y2):
            y2 = np.reshape(y2, np.shape(y1))

        return y1, y2


def _check_clf_targets(y_true, y_pred):
    """Check that y_true and y_pred belong to the same classification task

    This converts multiclass or binary types to a common shape, and raises a
    ValueError for a mix of multilabel and multiclass targets, a mix of
    multilabel formats, for the presence of continuous-valued or multioutput
    targets, or for targets of different lengths.

    Column vectors are squeezed to 1d.

    Parameters
    ----------
    y_true : array-like,

    y_pred : array-like

    Returns
    -------
    type_true : one of {'multilabel-indicator', 'multilabel-sequences', \
                        'multiclass', 'binary'}
        The type of the true target data, as output by
        ``utils.multiclass.type_of_target``

    y_true : array or indicator matrix or sequence of sequences

    y_pred : array or indicator matrix or sequence of sequences
    """
    y_true, y_pred = check_arrays(y_true, y_pred, allow_lists=True)
    type_true = type_of_target(y_true)
    type_pred = type_of_target(y_pred)

    if type_true.startswith('multilabel'):
        if not type_pred.startswith('multilabel'):
            raise ValueError("Can't handle mix of multilabel and multiclass "
                             "targets")
        if type_true != type_pred:
            raise ValueError("Can't handle mix of multilabel formats (label "
                             "indicator matrix and sequence of sequences)")
    elif type_pred.startswith('multilabel'):
        raise ValueError("Can't handle mix of multilabel and multiclass "
                         "targets")

    elif (type_pred in ('multiclass', 'binary')
          and type_true in ('multiclass', 'binary')):

        if 'multiclass' in (type_true, type_pred):
            # 'binary' can be removed
            type_true = type_pred = 'multiclass'

        y_true = np.ravel(y_true)
        y_pred = np.ravel(y_pred)

    else:
        raise ValueError("Can't handle %s/%s targets" % (type_true, type_pred))

    return type_true, y_true, y_pred


def auc(x, y, reorder=False):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`auc_score`.

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates.

    y : array, shape = [n]
        y coordinates.

    reorder : boolean, optional (default=False)
        If True, assume that the curve is ascending in the case of ties, as for
        an ROC curve. If the curve is non-ascending, the result will be wrong.

    Returns
    -------
    auc : float

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    >>> metrics.auc(fpr, tpr)
    0.75

    See also
    --------
    auc_score : Computes the area under the ROC curve

    """
    x, y = check_arrays(x, y)
    if x.shape[0] < 2:
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % x.shape)

    direction = 1
    if reorder:
        # reorder the data points according to the x axis and using y to
        # break ties
        order = np.lexsort((y, x))
        x, y = x[order], y[order]
    else:
        dx = np.diff(x)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
            else:
                raise ValueError("Reordering is not turned on, and "
                                 "the x array is not increasing: %s" % x)

    area = direction * np.trapz(y, x)

    return area


###############################################################################
# Binary classification loss
###############################################################################
def hinge_loss(y_true, pred_decision, pos_label=None, neg_label=None):
    """Average hinge loss (non-regularized)

    Assuming labels in y_true are encoded with +1 and -1, when a prediction
    mistake is made, ``margin = y_true * pred_decision`` is always negative
    (since the signs disagree), implying ``1 - margin`` is always greater than
    1.  The cumulated hinge loss is therefore an upper bound of the number of
    mistakes made by the classifier.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True target, consisting of integers of two values. The positive label
        must be greater than the negative label.

    pred_decision : array, shape = [n_samples] or [n_samples, n_classes]
        Predicted decisions, as output by decision_function (floats).

    Returns
    -------
    loss : float

    References
    ----------
    .. [1] `Wikipedia entry on the Hinge loss
            <http://en.wikipedia.org/wiki/Hinge_loss>`_

    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.metrics import hinge_loss
    >>> X = [[0], [1]]
    >>> y = [-1, 1]
    >>> est = svm.LinearSVC(random_state=0)
    >>> est.fit(X, y)
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
         random_state=0, tol=0.0001, verbose=0)
    >>> pred_decision = est.decision_function([[-2], [3], [0.5]])
    >>> pred_decision  # doctest: +ELLIPSIS
    array([-2.18...,  2.36...,  0.09...])
    >>> hinge_loss([-1, 1, 1], pred_decision)  # doctest: +ELLIPSIS
    0.30...

    """
    if pos_label is not None:
        warnings.warn("'pos_label' is deprecated and will be removed in "
                      "release 0.15.", DeprecationWarning)
    if neg_label is not None:
        warnings.warn("'neg_label' is unused and will be removed in "
                      "release 0.15.", DeprecationWarning)

    # TODO: multi-class hinge-loss

    # the rest of the code assumes that positive and negative labels
    # are encoded as +1 and -1 respectively
    if pos_label is not None:
        y_true = (np.asarray(y_true) == pos_label) * 2 - 1
    else:
        y_true = LabelBinarizer(neg_label=-1).fit_transform(y_true)[:, 0]

    margin = y_true * np.asarray(pred_decision)
    losses = 1 - margin
    # The hinge doesn't penalize good enough predictions.
    losses[losses <= 0] = 0
    return np.mean(losses)


###############################################################################
# Binary classification scores
###############################################################################
def average_precision_score(y_true, y_score):
    """Compute average precision (AP) from prediction scores

    This score corresponds to the area under the precision-recall curve.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    Returns
    -------
    average_precision : float

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <http://en.wikipedia.org/wiki/Average_precision>`_

    See also
    --------
    auc_score : Area under the ROC curve

    precision_recall_curve :
        Compute precision-recall pairs for different probability thresholds

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> average_precision_score(y_true, y_scores)  # doctest: +ELLIPSIS
    0.79...

    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


def auc_score(y_true, y_score):
    """Compute Area Under the Curve (AUC) from prediction scores

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    Returns
    -------
    auc : float

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    See also
    --------
    average_precision_score : Area under the precision-recall curve

    roc_curve : Compute Receiver operating characteristic (ROC)

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import auc_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> auc_score(y_true, y_scores)
    0.75

    """
    if len(np.unique(y_true)) != 2:
        raise ValueError("AUC is defined for binary classification only")
    fpr, tpr, tresholds = roc_curve(y_true, y_score)
    return auc(fpr, tpr, reorder=True)


def matthews_corrcoef(y_true, y_pred):
    """Compute the Matthews correlation coefficient (MCC) for binary classes

    The Matthews correlation coefficient is used in machine learning as a
    measure of the quality of binary (two-class) classifications. It takes into
    account true and false positives and negatives and is generally regarded as
    a balanced measure which can be used even if the classes are of very
    different sizes. The MCC is in essence a correlation coefficient value
    between -1 and +1. A coefficient of +1 represents a perfect prediction, 0
    an average random prediction and -1 an inverse prediction.  The statistic
    is also known as the phi coefficient. [source: Wikipedia]

    Only in the binary case does this relate to information about true and
    false positives and negatives. See references below.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    Returns
    -------
    mcc : float
        The Matthews correlation coefficient (+1 represents a perfect
        prediction, 0 an average random prediction and -1 and inverse
        prediction).

    References
    ----------
    .. [1] `Baldi, Brunak, Chauvin, Andersen and Nielsen, (2000). Assessing the
       accuracy of prediction algorithms for classification: an overview
       <http://dx.doi.org/10.1093/bioinformatics/16.5.412>`_

    .. [2] `Wikipedia entry for the Matthews Correlation Coefficient
       <http://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_

    Examples
    --------
    >>> from sklearn.metrics import matthews_corrcoef
    >>> y_true = [+1, +1, +1, -1]
    >>> y_pred = [+1, -1, +1, +1]
    >>> matthews_corrcoef(y_true, y_pred)  # doctest: +ELLIPSIS
    -0.33...

    """
    y_true, y_pred = check_arrays(y_true, y_pred)
    y_true, y_pred = _check_1d_array(y_true, y_pred, ravel=True)

    mcc = np.corrcoef(y_true, y_pred)[0, 1]
    if np.isnan(mcc):
        return 0.
    else:
        return mcc


def _binary_clf_curve(y_true, y_score, pos_label=None):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    pos_label : int, optional (default=1)
        The label of the positive class

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).

    tps : array, shape = [n_thresholds := len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    y_true, y_score = check_arrays(y_true, y_score)
    y_true, y_score = _check_1d_array(y_true, y_score, ravel=True)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.all(classes == [0, 1]) or
             np.all(classes == [-1, 1]) or
             np.all(classes == [0]) or
             np.all(classes == [-1]) or
             np.all(classes == [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # Sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = y_true.cumsum()[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def precision_recall_curve(y_true, probas_pred):
    """Compute precision-recall pairs for different probability thresholds

    Note: this implementation is restricted to the binary classification task.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The last precision and recall values are 1. and 0. respectively and do not
    have a corresponding threshold.  This ensures that the graph starts on the
    x axis.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification in range {-1, 1} or {0, 1}.

    probas_pred : array, shape = [n_samples]
        Estimated probabilities or decision function.

    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : array, shape = [n_thresholds := len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        precision and recall.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_curve
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> precision, recall, thresholds = precision_recall_curve(
    ...     y_true, y_scores)
    >>> precision  # doctest: +ELLIPSIS
    array([ 0.66...,  0.5       ,  1.        ,  1.        ])
    >>> recall
    array([ 1. ,  0.5,  0.5,  0. ])
    >>> thresholds
    array([ 0.35,  0.4 ,  0.8 ])

    """
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred)

    precision = tps / (tps + fps)
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def roc_curve(y_true, y_score, pos_label=None):
    """Compute Receiver operating characteristic (ROC)

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True binary labels in range {0, 1} or {-1, 1}.  If labels are not
        binary, pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or binary decisions.

    pos_label : int
        Label considered as positive and others are considered negative.

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing false positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute
        fpr and tpr.

    See also
    --------
    auc_score : Compute Area Under the Curve (AUC) from prediction scores

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <http://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])
    >>> tpr
    array([ 0.5,  0.5,  1. ,  1. ])
    >>> thresholds
    array([ 0.8 ,  0.4 ,  0.35,  0.1 ])

    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score, pos_label)

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] == 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless")
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] == 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless")
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


##############################################################################
# Multiclass general function
###############################################################################
def confusion_matrix(y_true, y_pred, labels=None):
    """Compute confusion matrix to evaluate the accuracy of a classification

    By definition a confusion matrix :math:`C` is such that :math:`C_{i, j}`
    is equal to the number of observations known to be in group :math:`i` but
    predicted to be in group :math:`j`.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.

    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.

    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix

    References
    ----------
    .. [1] `Wikipedia entry for the Confusion matrix
           <http://en.wikipedia.org/wiki/Confusion_matrix>`_

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])

    """
    y_true, y_pred = check_arrays(y_true, y_pred)
    y_true, y_pred = _check_1d_array(y_true, y_pred, ravel=True)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    n_labels = labels.size
    label_to_ind = dict((y, x) for x, y in enumerate(labels))
    # convert yt, yp into index
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])

    # intersect y_pred, y_true with labels, eliminate items not in labels
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    CM = np.asarray(
        coo_matrix(
            (np.ones(y_true.shape[0], dtype=np.int), (y_true, y_pred)),
            shape=(n_labels, n_labels)
        ).todense()
    )

    return CM


###############################################################################
# Multiclass loss function
###############################################################################
def zero_one_loss(y_true, y_pred, normalize=True):
    """Zero-one classification loss.

    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of misclassifications.
        Otherwise, return the fraction of misclassifications.

    Returns
    -------
    loss : float or int,
        If ``normalize == True``, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).

    Notes
    -----
    In multilabel classification, the zero_one_loss function corresponds to
    the subset zero-one loss: for each sample, the entire set of labels must be
    correctly predicted, otherwise the loss for that sample is equal to one.

    See also
    --------
    accuracy_score, hamming_loss, jaccard_similarity_score

    Examples
    --------
    >>> from sklearn.metrics import zero_one_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> zero_one_loss(y_true, y_pred)
    0.25
    >>> zero_one_loss(y_true, y_pred, normalize=False)
    1

    In the multilabel case with binary indicator format:

    >>> zero_one_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))
    0.5

    and with a list of labels format:

    >>> zero_one_loss([(1,), (3,)], [(1, 2), tuple()])
    1.0


    """
    y_true, y_pred = check_arrays(y_true, y_pred, allow_lists=True)
    score = accuracy_score(y_true, y_pred,
                           normalize=normalize)

    if normalize:
        return 1 - score
    else:
        if hasattr(y_true, "shape"):
            n_samples = (np.max(y_true.shape) if _is_1d(y_true)
                         else y_true.shape[0])

        else:
            n_samples = len(y_true)

        return n_samples - score


@deprecated("Function 'zero_one' has been renamed to "
            "'zero_one_loss' and will be removed in release 0.15."
            "Default behavior is changed from 'normalize=False' to "
            "'normalize=True'")
def zero_one(y_true, y_pred, normalize=False):
    """Zero-One classification loss

    If normalize is ``True``, return the fraction of misclassifications
    (float), else it returns the number of misclassifications (int). The best
    performance is 0.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    normalize : bool, optional (default=False)
        If ``False`` (default), return the number of misclassifications.
        Otherwise, return the fraction of misclassifications.

    Returns
    -------
    loss : float
        If normalize is True, return the fraction of misclassifications
        (float), else it returns the number of misclassifications (int).


    Examples
    --------
    >>> from sklearn.metrics import zero_one
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> zero_one(y_true, y_pred)
    1
    >>> zero_one(y_true, y_pred, normalize=True)
    0.25

    """
    return zero_one_loss(y_true, y_pred, normalize)


###############################################################################
# Multiclass score functions
###############################################################################

def jaccard_similarity_score(y_true, y_pred, normalize=True):
    """Jaccard similarity coefficient score

    The Jaccard index [1], or Jaccard similarity coefficient, defined as
    the size of the intersection divided by the size of the union of two label
    sets, is used to compare set of predicted labels for a sample to the
    corresponding set of labels in ``y_true``.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the sum of the Jaccard similarity coefficient
        over the sample set. Otherwise, return the average of Jaccard
        similarity coefficient.

    Returns
    -------
    score : float
        If ``normalize == True``, return the average Jaccard similarity
        coefficient, else it returns the sum of the Jaccard similarity
        coefficient over the sample set.

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also
    --------
    accuracy_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equivalent
    to the ``accuracy_score``. It differs in the multilabel classification
    problem.

    References
    ----------
    .. [1] `Wikipedia entry for the Jaccard index
           <http://en.wikipedia.org/wiki/Jaccard_index>`_


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import jaccard_similarity_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> jaccard_similarity_score(y_true, y_pred)
    0.5
    >>> jaccard_similarity_score(y_true, y_pred, normalize=False)
    2

    In the multilabel case with binary indicator format:

    >>> jaccard_similarity_score(np.array([[0.0, 1.0], [1.0, 1.0]]),\
        np.ones((2, 2)))
    0.75

    and with a list of labels format:

    >>> jaccard_similarity_score([(1,), (3,)], [(1, 2), tuple()])
    0.25

    """

    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)
    if y_type == 'multilabel-indicator':
        try:
            # oddly, we may get an "invalid" rather than a "divide"
            # error here
            old_err_settings = np.seterr(divide='ignore',
                                         invalid='ignore')
            y_pred_pos_label = y_pred == 1
            y_true_pos_label = y_true == 1
            pred_inter_true = np.sum(np.logical_and(y_pred_pos_label,
                                                    y_true_pos_label),
                                     axis=1)
            pred_union_true = np.sum(np.logical_or(y_pred_pos_label,
                                                   y_true_pos_label),
                                     axis=1)
            score = pred_inter_true / pred_union_true

            # If there is no label, it results in a Nan instead, we set
            # the jaccard to 1: lim_{x->0} x/x = 1
            # Note with py2.6 and np 1.3: we can't check safely for nan.
            score[pred_union_true == 0.0] = 1.0
        finally:
            np.seterr(**old_err_settings)

    elif y_type == 'multilabel-sequences':
        score = np.empty(len(y_true), dtype=np.float)
        for i, (true, pred) in enumerate(zip(y_pred, y_true)):
            true_set = set(true)
            pred_set = set(pred)
            size_true_union_pred = len(true_set | pred_set)
            # If there is no label, it results in a Nan instead, we set
            # the jaccard to 1: lim_{x->0} x/x = 1
            if size_true_union_pred == 0:
                score[i] = 1.
            else:
                score[i] = (len(true_set & pred_set) /
                            size_true_union_pred)
    else:
        score = y_true == y_pred

    if normalize:
        return np.mean(score)
    else:
        return np.sum(score)


def accuracy_score(y_true, y_pred, normalize=True):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    normalize : bool, optional (default=True)
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.

    Returns
    -------
    score : float
        If ``normalize == True``, return the correctly classified samples
        (float), else it returns the number of correctly classified samples
        (int).

        The best performance is 1 with ``normalize == True`` and the number
        of samples with ``normalize == False``.

    See also
    --------
    jaccard_similarity_score, hamming_loss, zero_one_loss

    Notes
    -----
    In binary and multiclass classification, this function is equal
    to the ``jaccard_similarity_score`` function.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import accuracy_score
    >>> y_pred = [0, 2, 1, 3]
    >>> y_true = [0, 1, 2, 3]
    >>> accuracy_score(y_true, y_pred)
    0.5
    >>> accuracy_score(y_true, y_pred, normalize=False)
    2

    In the multilabel case with binary indicator format:

    >>> accuracy_score(np.array([[0.0, 1.0], [1.0, 1.0]]), np.ones((2, 2)))
    0.5

    and with a list of labels format:

    >>> accuracy_score([(1,), (3,)], [(1, 2), tuple()])
    0.0

    """
    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)
    if y_type == 'multilabel-indicator':
        score = (y_pred != y_true).sum(axis=1) == 0
    elif y_type == 'multilabel-sequences':
        score = np.array([len(set(true) ^ set(pred)) == 0
                          for pred, true in zip(y_pred, y_true)])
    else:
        score = y_true == y_pred

    if normalize:
        return np.mean(score)
    else:
        return np.sum(score)


def f1_score(y_true, y_pred, labels=None, pos_label=1, average='weighted'):
    """Compute the F1 score, also known as balanced F-score or F-measure

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    In the multi-class and multi-label case, this is the weighted average of
    the F1 score of each class.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    labels : array
        Integer array of labels.

    pos_label : int, 1 by default
        If ``average`` is not ``None`` and the classification target is binary,
        only this class's scores will be returned.

    average : string, [None, 'micro', 'macro', 'samples', 'weighted' (default)]
        If ``None``, the scores for each class are returned. Otherwise,
        unless ``pos_label`` is given in binary classification, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).


    Returns
    -------
    f1_score : float or array of float, shape = [n_unique_labels]
        F1 score of the positive class in binary classification or weighted
        average of the F1 scores of each class for the multiclass task.

    References
    ----------
    .. [1] `Wikipedia entry for the F1-score
           <http://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics import f1_score
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> f1_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.666...

    In the multiclass case:

    >>> from sklearn.metrics import f1_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> f1_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.26...
    >>> f1_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.33...
    >>> f1_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    0.26...
    >>> f1_score(y_true, y_pred, average=None)
    array([ 0.8,  0. ,  0. ])

    In the multilabel case with binary indicator format:

    >>> from sklearn.metrics import f1_score
    >>> y_true = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> y_pred = np.ones((3, 3))
    >>> f1_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.59...
    >>> f1_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.61...
    >>> f1_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    0.65...
    >>> f1_score(y_true, y_pred, average='samples')  # doctest: +ELLIPSIS
    0.59...
    >>> f1_score(y_true, y_pred, average=None)
    array([ 0.5,  0.8,  0.5])

    and with a list of labels format:

    >>> from sklearn.metrics import f1_score
    >>> y_true = [(1, 2), (3,)]
    >>> y_pred = [(1, 2), tuple()]
    >>> f1_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.66...
    >>> f1_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.8...
    >>> f1_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    0.66...
    >>> f1_score(y_true, y_pred, average='samples')  # doctest: +ELLIPSIS
    0.5
    >>> f1_score(y_true, y_pred, average=None)
    array([ 1.,  1.,  0.])

    """
    return fbeta_score(y_true, y_pred, 1, labels=labels,
                       pos_label=pos_label, average=average)


def fbeta_score(y_true, y_pred, beta, labels=None, pos_label=1,
                average='weighted'):
    """Compute the F-beta score

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The `beta` parameter determines the weight of precision in the combined
    score. ``beta < 1`` lends more weight to precision, while ``beta > 1``
    favors precision (``beta == 0`` considers only precision, ``beta == inf``
    only recall).

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    beta: float
        Weight of precision in harmonic mean.

    labels : array
        Integer array of labels.

    pos_label : int, 1 by default
        If ``average`` is not ``None`` and the classification target is binary,
        only this class's scores will be returned.

    average : string, [None, 'micro', 'macro', 'samples', 'weighted' (default)]
        If ``None``, the scores for each class are returned. Otherwise,
        unless ``pos_label`` is given in binary classification, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    Returns
    -------
    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score of the positive class in binary classification or weighted
        average of the F-beta score of each class for the multiclass task.

    References
    ----------
    .. [1] R. Baeza-Yates and B. Ribeiro-Neto (2011).
           Modern Information Retrieval. Addison Wesley, pp. 327-328.

    .. [2] `Wikipedia entry for the F1-score
           <http://en.wikipedia.org/wiki/F1_score>`_

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics import fbeta_score
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> fbeta_score(y_true, y_pred, beta=0.5)  # doctest: +ELLIPSIS
    0.83...
    >>> fbeta_score(y_true, y_pred, beta=1)  # doctest: +ELLIPSIS
    0.66...
    >>> fbeta_score(y_true, y_pred, beta=2)  # doctest: +ELLIPSIS
    0.55...

    In the multiclass case:

    >>> from sklearn.metrics import fbeta_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.23...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.33...
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.23...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    ... # doctest: +ELLIPSIS
    array([ 0.71...,  0.        ,  0.        ])


    In the multilabel case with binary indicator format:

    >>> from sklearn.metrics import fbeta_score
    >>> y_true = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> y_pred = np.ones((3, 3))
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.49...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    0.5
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.54...
    >>> fbeta_score(y_true, y_pred, average='samples', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.66...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    ... # doctest: +ELLIPSIS
    array([ 0.38...,  0.71...,  0.38...])

    and with a list of labels format:

    >>> from sklearn.metrics import fbeta_score
    >>> y_true = [(1, 2), (3,)]
    >>> y_pred = [(1, 2), tuple()]
    >>> fbeta_score(y_true, y_pred, average='macro', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.66...
    >>> fbeta_score(y_true, y_pred, average='micro', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.90...
    >>> fbeta_score(y_true, y_pred, average='weighted', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.66...
    >>> fbeta_score(y_true, y_pred, average='samples', beta=0.5)
    ... # doctest: +ELLIPSIS
    0.42...
    >>> fbeta_score(y_true, y_pred, average=None, beta=0.5)
    array([ 1.,  1.,  0.])

    """
    _, _, f, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 beta=beta,
                                                 labels=labels,
                                                 pos_label=pos_label,
                                                 average=average)
    return f


def _tp_tn_fp_fn(y_true, y_pred, labels=None):
    """Compute the number of true/false positives/negative for each class

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    labels : array, shape = [n_labels], optional
        Integer array of labels.

    Returns
    -------
    true_pos : array of int, shape = [n_unique_labels]
        Number of true positives

    true_neg : array of int, shape = [n_unique_labels]
        Number of true negative

    false_pos : array of int, shape = [n_unique_labels]
        Number of false positives

    false_pos : array of int, shape = [n_unique_labels]
        Number of false positives

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics.metrics import _tp_tn_fp_fn
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> _tp_tn_fp_fn(y_true, y_pred)
    (array([2, 1]), array([1, 2]), array([1, 0]), array([0, 1]))

    In the multiclass case:
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 2, 1, 0, 0, 1])
    >>> _tp_tn_fp_fn(y_true, y_pred)
    (array([2, 0, 0]), array([3, 2, 3]), array([1, 2, 1]), array([0, 2, 2]))

    In the multilabel case with binary indicator format:

    >>> _tp_tn_fp_fn(np.array([[0.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2)))
    (array([0, 0]), array([1, 0]), array([0, 0]), array([1, 2]))

    and with a list of labels format:

    >>> _tp_tn_fp_fn([(1, 2), (3,)], [(1, 2), tuple()])  # doctest: +ELLIPSIS
    (array([1, 1, 0]), array([1, 1, 1]), array([0, 0, 0]), array([0, 0, 1]))

    """
    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)
    n_labels = labels.size
    true_pos = np.zeros((n_labels), dtype=np.int)
    false_pos = np.zeros((n_labels), dtype=np.int)
    false_neg = np.zeros((n_labels), dtype=np.int)

    if y_type == 'multilabel-indicator':
        true_pos = np.sum(np.logical_and(y_true == 1,
                                         y_pred == 1), axis=0)
        false_pos = np.sum(np.logical_and(y_true != 1,
                                          y_pred == 1), axis=0)
        false_neg = np.sum(np.logical_and(y_true == 1,
                                          y_pred != 1), axis=0)

    elif y_type == 'multilabel-sequences':
        idx_to_label = dict((label_i, i)
                            for i, label_i in enumerate(labels))

        for true, pred in zip(y_true, y_pred):
            true_set = np.array([idx_to_label[l] for l in set(true)],
                                dtype=np.int)
            pred_set = np.array([idx_to_label[l] for l in set(pred)],
                                dtype=np.int)
            true_pos[np.intersect1d(true_set, pred_set)] += 1
            false_pos[np.setdiff1d(pred_set, true_set)] += 1
            false_neg[np.setdiff1d(true_set, pred_set)] += 1

    else:
        for i, label_i in enumerate(labels):
            true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
            false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
            false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)

    # Compute the true_neg using the tp, fp and fn
    if hasattr(y_true, "shape"):
        n_samples = (np.max(y_true.shape) if _is_1d(y_true)
                     else y_true.shape[0])
    else:
        n_samples = len(y_true)

    true_neg = n_samples - true_pos - false_pos - false_neg

    return true_pos, true_neg, false_pos, false_neg


def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None,
                                    pos_label=1, average=None):
    """Compute precision, recall, F-measure and support for each class

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.

    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    beta : float, 1.0 by default
        The strength of recall versus precision in the F-score.

    labels : array
        Integer array of labels.

    pos_label : int, 1 by default
        If ``average`` is not ``None`` and the classification target is binary,
        only this class's scores will be returned.

    average : string, [None (default), 'micro', 'macro', 'samples', 'weighted']
        If ``None``, the scores for each class are returned. Otherwise,
        unless ``pos_label`` is given in binary classification, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).


    Returns
    -------
    precision: float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    recall: float (if average is not None) or array of float, , shape =\
        [n_unique_labels]

    fbeta_score: float (if average is not None) or array of float, shape =\
        [n_unique_labels]

    support: int (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <http://en.wikipedia.org/wiki/Precision_and_recall>`_

    .. [2] `Wikipedia entry for the F1-score
           <http://en.wikipedia.org/wiki/F1_score>`_

    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> p, r, f, s = precision_recall_fscore_support(y_true, y_pred, beta=0.5)
    >>> p  # doctest: +ELLIPSIS
    array([ 0.66...,  1.        ])
    >>> r
    array([ 1. ,  0.5])
    >>> f  # doctest: +ELLIPSIS
    array([ 0.71...,  0.83...])
    >>> s  # doctest: +ELLIPSIS
    array([2, 2]...)

    In the multiclass case:

    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 2, 1, 0, 0, 1])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    ... # doctest: +ELLIPSIS
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    ... # doctest: +ELLIPSIS
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    (0.22..., 0.33..., 0.26..., None)

    In the multilabel case with binary indicator format:

    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> y_pred = np.ones((3, 3))
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    ... # doctest: +ELLIPSIS
    (0.44..., 1.0, 0.59..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    ... # doctest: +ELLIPSIS
    (0.44..., 1.0, 0.61..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    (0.499..., 1.0, 0.65..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='samples')
    ... # doctest: +ELLIPSIS
    (1.0, 0.44..., 0.59..., None)

    and with a list of labels format:

    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = [(1, 2), (3,)]
    >>> y_pred = [(1, 2), tuple()]
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    ... # doctest: +ELLIPSIS
    (0.66..., 0.66..., 0.66..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    ... # doctest: +ELLIPSIS
    (1.0, 0.66..., 0.8..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    (0.66..., 0.66..., 0.66..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='samples')
    ... # doctest: +ELLIPSIS
    (0.5, 1.0, 0.5, None)

    """
    if beta <= 0:
        raise ValueError("beta should be >0 in the F-beta score")
    beta2 = beta ** 2

    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if average == "samples":
        if y_type == 'multilabel-indicator':
            y_true_pos_label = y_true == 1
            y_pred_pos_label = y_pred == 1
            size_inter = np.sum(np.logical_and(y_true_pos_label,
                                               y_pred_pos_label), axis=1)
            size_true = np.sum(y_true_pos_label, axis=1)
            size_pred = np.sum(y_pred_pos_label, axis=1)

        elif y_type == 'multilabel-sequences':
            size_inter = np.empty(len(y_true), dtype=np.int)
            size_true = np.empty(len(y_true), dtype=np.int)
            size_pred = np.empty(len(y_true), dtype=np.int)
            for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                true_set = set(true)
                pred_set = set(pred)
                size_inter[i] = len(true_set & pred_set)
                size_pred[i] = len(pred_set)
                size_true[i] = len(true_set)
        else:
            raise ValueError("Example-based precision, recall, fscore is "
                             "not meaning full outside multilabe"
                             "classification. See the accuracy_score instead.")

        try:
            # oddly, we may get an "invalid" rather than a "divide" error
            # here
            old_err_settings = np.seterr(divide='ignore', invalid='ignore')

            precision = size_inter / size_true
            recall = size_inter / size_pred
            f_score = ((1 + beta2 ** 2) * size_inter /
                       (beta2 * size_pred + size_true))
        finally:
            np.seterr(**old_err_settings)

        precision[size_true == 0] = 1.0
        recall[size_pred == 0] = 1.0
        f_score[(beta2 * size_pred + size_true) == 0] = 1.0

        precision = np.mean(precision)
        recall = np.mean(recall)
        f_score = np.mean(f_score)

        return precision, recall, f_score, None

    true_pos, _, false_pos, false_neg = _tp_tn_fp_fn(y_true, y_pred, labels)
    support = true_pos + false_neg

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        # precision and recall
        precision = divide(true_pos.astype(np.float), true_pos + false_pos)
        recall = divide(true_pos.astype(np.float), true_pos + false_neg)

        # handle division by 0 in precision and recall
        precision[(true_pos + false_pos) == 0] = 0.0
        recall[(true_pos + false_neg) == 0] = 0.0

        # fbeta score
        fscore = divide((1 + beta2) * precision * recall,
                        beta2 * precision + recall)

        # handle division by 0 in fscore
        fscore[(beta2 * precision + recall) == 0] = 0.0
    finally:
        np.seterr(**old_err_settings)

    if not average:
        return precision, recall, fscore, support

    elif y_type == 'binary' and pos_label is not None:
        if pos_label not in labels:
            if len(labels) == 1:
                # Only negative labels
                return (0., 0., 0., 0)
            raise ValueError("pos_label=%d is not a valid label: %r" %
                             (pos_label, labels))
        pos_label_idx = list(labels).index(pos_label)
        return (precision[pos_label_idx], recall[pos_label_idx],
                fscore[pos_label_idx], support[pos_label_idx])
    else:
        average_options = (None, 'micro', 'macro', 'weighted', 'samples')
        if average == 'micro':
            avg_precision = divide(true_pos.sum(),
                                   true_pos.sum() + false_pos.sum(),
                                   dtype=np.double)
            avg_recall = divide(true_pos.sum(),
                                true_pos.sum() + false_neg.sum(),
                                dtype=np.double)
            avg_fscore = divide((1 + beta2) * (avg_precision * avg_recall),
                                beta2 * avg_precision + avg_recall,
                                dtype=np.double)

            if np.isnan(avg_precision):
                avg_precision = 0.

            if np.isnan(avg_recall):
                avg_recall = 0.

            if np.isnan(avg_fscore):
                avg_fscore = 0.

        elif average == 'macro':
            avg_precision = np.mean(precision)
            avg_recall = np.mean(recall)
            avg_fscore = np.mean(fscore)

        elif average == 'weighted':
            if np.all(support == 0):
                avg_precision = 0.
                avg_recall = 0.
                avg_fscore = 0.
            else:
                avg_precision = np.average(precision, weights=support)
                avg_recall = np.average(recall, weights=support)
                avg_fscore = np.average(fscore, weights=support)

        else:
            raise ValueError('average has to be one of ' +
                             str(average_options))

        return avg_precision, avg_recall, avg_fscore, None


def precision_score(y_true, y_pred, labels=None, pos_label=1,
                    average='weighted'):
    """Compute the precision

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    labels : array
        Integer array of labels.

    pos_label : int, 1 by default
        If ``average`` is not ``None`` and the classification target is binary,
        only this class's scores will be returned.

    average : string, [None, 'micro', 'macro', 'samples', 'weighted' (default)]
        If ``None``, the scores for each class are returned. Otherwise,
        unless ``pos_label`` is given in binary classification, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision of the positive class in binary classification or weighted
        average of the precision of each class for the multiclass task.

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics import precision_score
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> precision_score(y_true, y_pred)
    1.0

    In the multiclass case:

    >>> from sklearn.metrics import precision_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> precision_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.22...
    >>> precision_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.33...
    >>> precision_score(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    0.22...
    >>> precision_score(y_true, y_pred, average=None)  # doctest: +ELLIPSIS
    array([ 0.66...,  0.        ,  0.        ])

    In the multilabel case with binary indicator format:

    >>> from sklearn.metrics import precision_score
    >>> y_true = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> y_pred = np.ones((3, 3))
    >>> precision_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.44...
    >>> precision_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.44...
    >>> precision_score(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    0.49...
    >>> precision_score(y_true, y_pred, average='samples')
    1.0
    >>> precision_score(y_true, y_pred, average=None)
    ... # doctest: +ELLIPSIS
    array([ 0.33...,  0.66...,  0.33...])

    and with a list of labels format:

    >>> from sklearn.metrics import precision_score
    >>> y_true = [(1, 2), (3,)]
    >>> y_pred = [(1, 2), tuple()]
    >>> precision_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.66...
    >>> precision_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    1.0
    >>> precision_score(y_true, y_pred, average='weighted')
    ... # doctest: +ELLIPSIS
    0.66...
    >>> precision_score(y_true, y_pred, average='samples')
    ... # doctest: +ELLIPSIS
    0.5
    >>> precision_score(y_true, y_pred, average=None)
    array([ 1.,  1.,  0.])


    """
    p, _, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=labels,
                                                 pos_label=pos_label,
                                                 average=average)
    return p


def recall_score(y_true, y_pred, labels=None, pos_label=1, average='weighted'):
    """Compute the recall

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    labels : array
        Integer array of labels.

    pos_label : int, 1 by default
        If ``average`` is not ``None`` and the classification target is binary,
        only this class's scores will be returned.

    average : string, [None, 'micro', 'macro', 'samples', 'weighted' (default)]
        If ``None``, the scores for each class are returned. Otherwise,
        unless ``pos_label`` is given in binary classification, this
        determines the type of averaging performed on the data:

        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall of the positive class in binary classification or weighted
        average of the recall of each class for the multiclass task.

    Examples
    --------
    In the binary case:

    >>> from sklearn.metrics import recall_score
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 0, 1]
    >>> recall_score(y_true, y_pred)
    0.5

    In the multiclass case:

    >>> from sklearn.metrics import recall_score
    >>> y_true = [0, 1, 2, 0, 1, 2]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> recall_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.33...
    >>> recall_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.33...
    >>> recall_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    0.33...
    >>> recall_score(y_true, y_pred, average=None)
    array([ 1.,  0.,  0.])

    In the multilabel case with binary indicator format:

    >>> from sklearn.metrics import recall_score
    >>> y_true = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> y_pred = np.ones((3, 3))
    >>> recall_score(y_true, y_pred, average='macro')
    1.0
    >>> recall_score(y_true, y_pred, average='micro')
    1.0
    >>> recall_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    1.0
    >>> recall_score(y_true, y_pred, average='samples')  # doctest: +ELLIPSIS
    0.44...
    >>> recall_score(y_true, y_pred, average=None)
    array([ 1.,  1.,  1.])

    and with a list of labels format:

    >>> from sklearn.metrics import recall_score
    >>> y_true = [(1, 2), (3,)]
    >>> y_pred = [(1, 2), tuple()]
    >>> recall_score(y_true, y_pred, average='macro')  # doctest: +ELLIPSIS
    0.66...
    >>> recall_score(y_true, y_pred, average='micro')  # doctest: +ELLIPSIS
    0.66...
    >>> recall_score(y_true, y_pred, average='weighted')  # doctest: +ELLIPSIS
    0.66...
    >>> recall_score(y_true, y_pred, average='samples')
    1.0
    >>> recall_score(y_true, y_pred, average=None)
    array([ 1.,  1.,  0.])
    """
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred,
                                                 labels=labels,
                                                 pos_label=pos_label,
                                                 average=average)
    return r


@deprecated("Function zero_one_score has been renamed to "
            'accuracy_score'" and will be removed in release 0.15.")
def zero_one_score(y_true, y_pred):
    """Zero-one classification score (accuracy)

    Parameters
    ----------
    y_true : array-like, shape = n_samples
        Ground truth (correct) labels.

    y_pred : array-like, shape = n_samples
        Predicted labels, as returned by a classifier.

    Returns
    -------
    score : float
        Fraction of correct predictions in ``y_pred``. The best performance is
        1.

    """
    return accuracy_score(y_true, y_pred)


###############################################################################
# Multiclass utility function
###############################################################################
def classification_report(y_true, y_pred, labels=None, target_names=None):
    """Build a text report showing the main classification metrics

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) target values.

    y_pred : array-like or list of labels or label indicator matrix
        Estimated targets as returned by a classifier.

    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.

    target_names : list of strings
        Optional display names matching the labels (same order).

    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.

    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 0]
    >>> y_pred = [0, 0, 2, 2, 0]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    <BLANKLINE>
        class 0       0.67      1.00      0.80         2
        class 1       0.00      0.00      0.00         1
        class 2       1.00      1.00      1.00         2
    <BLANKLINE>
    avg / total       0.67      0.80      0.72         5
    <BLANKLINE>

    """

    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    last_line_heading = 'avg / total'

    if target_names is None:
        width = len(last_line_heading)
        target_names = ['%d' % l for l in labels]
    else:
        width = max(len(cn) for cn in target_names)
        width = max(width, len(last_line_heading))

    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'

    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None)

    for i, label in enumerate(labels):
        values = [target_names[i]]
        for v in (p[i], r[i], f1[i]):
            values += ["%0.2f" % float(v)]
        values += ["%d" % int(s[i])]
        report += fmt % tuple(values)

    report += '\n'

    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["%0.2f" % float(v)]
    values += ['%d' % np.sum(s)]
    report += fmt % tuple(values)
    return report


###############################################################################
# Multilabel loss function
###############################################################################
def hamming_loss(y_true, y_pred, classes=None):
    """Compute the average Hamming loss.

    The Hamming loss is the fraction of labels that are incorrectly predicted.

    Parameters
    ----------
    y_true : array-like or list of labels or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or list of labels or label indicator matrix
        Predicted labels, as returned by a classifier.

    classes : array, shape = [n_labels], optional
        Integer array of labels.

    Returns
    -------
    loss : float or int,
        Return the average Hamming loss between element of ``y_true`` and
        ``y_pred``.

    See Also
    --------
    accuracy_score, jaccard_similarity_score, zero_one_loss

    Notes
    -----
    In multiclass classification, the Hamming loss correspond to the Hamming
    distance between ``y_true`` and ``y_pred`` which is equivalent to the
    subset ``zero_one_loss`` function.

    In multilabel classification, the Hamming loss is different from the
    subset zero-one loss. The zero-one loss considers the entire set of labels
    for a given sample incorrect if it does entirely match the true set of
    labels. Hamming loss is more forgiving in that it penalizes the individual
    labels.

    The Hamming loss is upperbounded by the subset zero-one loss. When
    normalized over samples, the Hamming loss is always between 0 and 1.

    References
    ----------
    .. [1] Grigorios Tsoumakas, Ioannis Katakis. Multi-Label Classification:
           An Overview. International Journal of Data Warehousing & Mining,
           3(3), 1-13, July-September 2007.

    .. [2] `Wikipedia entry on the Hamming distance
           <http://en.wikipedia.org/wiki/Hamming_distance>`_

    Examples
    --------
    >>> from sklearn.metrics import hamming_loss
    >>> y_pred = [1, 2, 3, 4]
    >>> y_true = [2, 2, 3, 4]
    >>> hamming_loss(y_true, y_pred)
    0.25

    In the multilabel case with binary indicator format:

    >>> hamming_loss(np.array([[0.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2)))
    0.75

    and with a list of labels format:

    >>> hamming_loss([(1, 2), (3,)], [(1, 2), tuple()])  # doctest: +ELLIPSIS
    0.166...

    """
    y_type, y_true, y_pred = _check_clf_targets(y_true, y_pred)
    if classes is None:
        classes = unique_labels(y_true, y_pred)
    else:
        classes = np.asarray(classes)

    if y_type == 'multilabel-indicator':
        return np.mean(y_true != y_pred)
    elif y_type == 'multilabel-sequences':
            loss = np.array([len(set(pred) ^ set(true))
                             for pred, true in zip(y_pred, y_true)])

            return np.mean(loss) / np.size(classes)

    else:
        return sp_hamming(y_true, y_pred)


###############################################################################
# Regression loss functions
###############################################################################
def mean_absolute_error(y_true, y_pred):
    """Mean absolute error regression loss

    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    loss : float
        A positive floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> mean_absolute_error(y_true, y_pred)
    0.75

    """
    y_true, y_pred = check_arrays(y_true, y_pred)

    # Handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs(y_pred - y_true))


def mean_squared_error(y_true, y_pred):
    """Mean squared error regression loss

    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    loss : float
        A positive floating point value (the best value is 0.0).

    Examples
    --------
    >>> from sklearn.metrics import mean_squared_error
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    >>> y_true = [[0.5, 1],[-1, 1],[7, -6]]
    >>> y_pred = [[0, 2],[-1, 2],[8, -5]]
    >>> mean_squared_error(y_true, y_pred)  # doctest: +ELLIPSIS
    0.708...

    """
    y_true, y_pred = check_arrays(y_true, y_pred)

    # Handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean((y_pred - y_true) ** 2)


###############################################################################
# Regression score functions
###############################################################################
def explained_variance_score(y_true, y_pred):
    """Explained variance regression score function

    Best possible score is 1.0, lower values are worse.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values.

    Returns
    -------
    score : float
        The explained variance.

    Notes
    -----
    This is not a symmetric function.

    Examples
    --------
    >>> from sklearn.metrics import explained_variance_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> explained_variance_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.957...

    """
    y_true, y_pred = check_arrays(y_true, y_pred)

    # Handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred)

    numerator = np.var(y_true - y_pred)
    denominator = np.var(y_true)
    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            # arbitrary set to zero to avoid -inf scores, having a constant
            # y_true is not interesting for scoring a regression anyway
            return 0.0
    return 1 - numerator / denominator


def r2_score(y_true, y_pred):
    """R?? (coefficient of determination) regression score function.

    Best possible score is 1.0, lower values are worse.

    Parameters
    ----------
    y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    z : float
        The R?? score.

    Notes
    -----
    This is not a symmetric function.

    Unlike most other scores, R?? score may be negative (it need not actually
    be the square of a quantity R).

    References
    ----------
    .. [1] `Wikipedia entry on the Coefficient of determination
            <http://en.wikipedia.org/wiki/Coefficient_of_determination>`_

    Examples
    --------
    >>> from sklearn.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.948...
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> r2_score(y_true, y_pred)  # doctest: +ELLIPSIS
    0.938...

    """
    y_true, y_pred = check_arrays(y_true, y_pred)

    # Handle mix 1d representation
    if _is_1d(y_true):
        y_true, y_pred = _check_1d_array(y_true, y_pred, ravel=True)

    if len(y_true) == 1:
        raise ValueError("r2_score can only be computed given more than one"
                         " sample.")
    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean(axis=0)) ** 2).sum()

    if denominator == 0.0:
        if numerator == 0.0:
            return 1.0
        else:
            # arbitrary set to zero to avoid -inf scores, having a constant
            # y_true is not interesting for scoring a regression anyway
            return 0.0

    return 1 - numerator / denominator
