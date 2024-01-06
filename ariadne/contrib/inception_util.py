from typing import Optional

from cassis import Cas
from cassis.typesystem import FeatureStructure

SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
IS_PREDICTION = "inception_internal_predicted"
FEATURE_NAME_SCORE_SUFFIX = "_score"
FEATURE_NAME_SCORE_EXPLANATION_SUFFIX = "_score_explanation"
FEATURE_NAME_AUTO_ACCEPT_MODE_SUFFIX = "_auto_accept"


def create_prediction(
    cas: Cas,
    layer: str,
    feature: str,
    begin: int,
    end: int,
    label: str,
    score: Optional[int] = None,
    score_explanation: Optional[str] = None,
    auto_accept: Optional[bool] = None,
) -> FeatureStructure:
    """
    Create a prediction

    :param cas: the annotated document
    :param layer: the layer on which to create the prediction
    :param feature: the feature to predict
    :param begin: the offset of the first character of the prediction
    :param end: the offset of the first character after the prediction
    :param label: the predicted label
    :param score: the score
    :param score_explanation: a rationale for the score / prediction
    :param auto_accept: whether the prediction should be automatically accepted
    :return: the prediction annotation
    """
    AnnotationType = cas.typesystem.get_type(layer)

    fields = {"begin": begin, "end": end, IS_PREDICTION: True, feature: label}
    prediction = AnnotationType(**fields)

    if score is not None:
        prediction[f"{feature}{FEATURE_NAME_SCORE_SUFFIX}"] = score

    if score_explanation is not None:
        prediction[f"{feature}{FEATURE_NAME_SCORE_EXPLANATION_SUFFIX}"] = score_explanation

    if auto_accept is not None:
        prediction[f"{feature}{FEATURE_NAME_AUTO_ACCEPT_MODE_SUFFIX}"] = auto_accept

    return prediction
