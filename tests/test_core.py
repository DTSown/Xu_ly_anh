"""Unit and cross-library tests required by the project acceptance checklist."""

import cv2
import numpy as np
import pytest

from core.color import downscale_mean, rgb_to_gray
from core.filters import convolve2d, gaussian_blur, gaussian_kernel
from core.morphology import closing, dilate, erode, rect_se
from core.segmentation import (
    connected_components_labeling,
    filter_boxes,
    get_bounding_boxes,
    detect_diagonal_gutters,
    sort_reading_order,
    suppress_nested,
)
from core.threshold import otsu_threshold
from main import crop_panel, process
from pipeline import segment_page


def component_count(labels):
    return len(np.unique(labels)) - (1 if np.any(labels == 0) else 0)


def test_rgb_to_gray_bgr_weights():
    image = np.array([[[10, 20, 30], [255, 0, 0]]], dtype=np.uint8)
    expected = np.array(
        [[0.114 * 10 + 0.587 * 20 + 0.299 * 30, 0.114 * 255]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(rgb_to_gray(image), expected, atol=1e-12)


def test_rgb_to_gray_accepts_2d_image():
    image = np.array([[0, 127], [200, 255]], dtype=np.uint8)
    result = rgb_to_gray(image)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, image.astype(np.float64))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 2.0])
def test_gaussian_kernel_normalized_and_symmetric(sigma):
    kernel = gaussian_kernel(sigma)
    assert np.isclose(kernel.sum(), 1.0, atol=1e-14)
    np.testing.assert_allclose(kernel, kernel[::-1, :], atol=1e-14)
    np.testing.assert_allclose(kernel, kernel[:, ::-1], atol=1e-14)


def test_gaussian_blur_matches_full_2d_convolution():
    rng = np.random.default_rng(20261009)
    image = rng.normal(size=(17, 19))
    sigma = 1.3
    actual = gaussian_blur(image, sigma)
    expected = convolve2d(image, gaussian_kernel(sigma), pad_mode="edge")
    max_error = float(np.max(np.abs(actual - expected)))
    print(f"gaussian_separable_vs_2d_max_abs_error={max_error:.3e}")
    assert max_error < 1e-6


def test_convolve2d_identity():
    image = np.arange(20, dtype=np.float64).reshape(4, 5)
    actual = convolve2d(image, np.array([[1.0]]))
    np.testing.assert_array_equal(actual, image)


def test_convolve2d_mean_3x3_hand_calculated():
    image = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
    expected = np.array(
        [[7 / 3, 3, 11 / 3], [13 / 3, 5, 17 / 3], [19 / 3, 7, 23 / 3]]
    )
    actual = convolve2d(image, np.ones((3, 3)) / 9)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_otsu_bimodal_threshold_separates_peaks():
    image = np.array([20] * 100 + [220] * 100, dtype=np.uint8).reshape(20, 10)
    threshold = otsu_threshold(image)
    assert 20 <= threshold < 220


@pytest.mark.parametrize("value", [0, 73, 255])
def test_otsu_constant_image_does_not_crash(value):
    image = np.full((11, 13), value, dtype=np.uint8)
    threshold = otsu_threshold(image)
    assert isinstance(threshold, int)
    assert 0 <= threshold <= 255


def test_morphology_closing_connects_two_pixel_gap():
    image = np.zeros((9, 13), dtype=bool)
    image[3:6, 2:5] = True
    image[3:6, 7:10] = True
    result = closing(image, rect_se(3))
    assert result[4, 5]
    assert result[4, 6]


def test_morphology_order_extensivity_and_closing_idempotence():
    image = np.zeros((13, 15), dtype=bool)
    image[2:11, 3:12] = True
    image[5:8, 6:8] = False
    se = rect_se(3)
    eroded = erode(image, se)
    dilated = dilate(image, se)
    closed = closing(image, se)
    assert np.all(eroded <= image)
    assert np.all(image <= dilated)
    assert np.all(image <= closed)
    np.testing.assert_array_equal(closing(closed, se), closed)


def test_ccl_two_separate_regions():
    image = np.zeros((8, 9), dtype=bool)
    image[1:3, 1:3] = True
    image[5:7, 6:8] = True
    assert component_count(connected_components_labeling(image)) == 2


def test_ccl_diagonal_touch_is_one_component():
    image = np.zeros((5, 5), dtype=bool)
    image[1, 1] = True
    image[2, 2] = True
    assert component_count(connected_components_labeling(image)) == 1


def test_ccl_u_shape_is_one_component():
    image = np.zeros((8, 8), dtype=bool)
    image[1:7, 1] = True
    image[1:7, 6] = True
    image[6, 1:7] = True
    assert component_count(connected_components_labeling(image)) == 1


def test_ccl_empty_image():
    labels = connected_components_labeling(np.zeros((6, 7), dtype=bool))
    assert labels.shape == (6, 7)
    assert labels.dtype == np.int32
    assert component_count(labels) == 0


def test_get_bounding_boxes():
    labels = np.zeros((8, 10), dtype=np.int32)
    labels[1:4, 2:6] = 4
    labels[5:7, 7:9] = 9
    assert get_bounding_boxes(labels) == [
        {"box": (2, 1, 5, 3), "area": 12},
        {"box": (7, 5, 8, 6), "area": 4},
    ]


def test_filter_boxes_uses_inclusive_bbox_area():
    boxes = [
        {"box": (0, 0, 0, 0), "area": 1},
        {"box": (0, 0, 4, 4), "area": 25},
        {"box": (0, 0, 9, 9), "area": 100},
    ]
    assert filter_boxes(boxes, 10, 10, min_frac=0.20, max_frac=0.80) == [(0, 0, 4, 4)]


def test_suppress_nested_preserves_adjacent_panels():
    outer = (0, 0, 99, 99)
    nested = (10, 10, 30, 30)
    adjacent = (101, 0, 180, 99)
    touching = (181, 0, 260, 99)
    assert suppress_nested([outer, nested, adjacent, touching]) == [outer, adjacent, touching]


def test_sort_reading_order_staggered_layout():
    top_right = (100, 10, 180, 80)
    bottom_left = (10, 120, 90, 200)
    top_left = (5, 20, 85, 90)
    bottom_right = (110, 130, 190, 210)
    assert sort_reading_order([bottom_right, top_right, bottom_left, top_left]) == [
        top_left,
        top_right,
        bottom_left,
        bottom_right,
    ]


def test_sort_reading_order_same_row_with_unequal_heights():
    right_tall = (396, 738, 713, 1051)
    left_short = (66, 822, 255, 961)
    assert sort_reading_order([right_tall, left_short]) == [left_short, right_tall]


def test_detect_diagonal_gutters_and_mask_panel():
    gray = np.full((180, 120), 205.0)
    x = np.arange(gray.shape[1])
    expected = [(-0.2, 65), (-0.2, 145)]
    for a, b in expected:
        y = np.rint(a * x + b).astype(int)
        for offset in range(-3, 4):
            yy = np.clip(y + offset, 0, gray.shape[0] - 1)
            gray[yy, x] = 255
        gray[np.clip(y + 5, 0, gray.shape[0] - 1), x] = 0
    gutters = detect_diagonal_gutters(gray)
    assert len(gutters) == 2
    assert all(abs(a + 0.2) <= 0.03 for a, _ in gutters)

    image = np.zeros((180, 120, 3), dtype=np.uint8)
    crop = crop_panel(image, (0, 0, 119, 70), 0, gutters)
    assert np.any(crop == 255)
    assert np.any(crop == 0)


def test_against_opencv_and_scipy_references():
    scipy_signal = pytest.importorskip("scipy.signal")
    rng = np.random.default_rng(3148)
    image = rng.integers(0, 256, size=(41, 37), dtype=np.uint8)
    sigma = 1.5
    kernel = gaussian_kernel(sigma)

    ours_gaussian = gaussian_blur(image.astype(np.float64), sigma)
    scipy_gaussian = scipy_signal.convolve2d(
        image.astype(np.float64), kernel, mode="same", boundary="symm"
    )
    cv_gaussian = cv2.GaussianBlur(
        image.astype(np.float64),
        (kernel.shape[1], kernel.shape[0]),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
    scipy_error = float(np.max(np.abs(ours_gaussian - scipy_gaussian)))
    cv_error = float(np.max(np.abs(ours_gaussian - cv_gaussian)))

    bimodal = np.vstack(
        [rng.integers(10, 61, size=(32, 64)), rng.integers(180, 241, size=(32, 64))]
    ).astype(np.uint8)
    ours_otsu = otsu_threshold(bimodal)
    cv_otsu, _ = cv2.threshold(bimodal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_difference = abs(ours_otsu - int(cv_otsu))

    binary = rng.random((39, 43)) > 0.72
    se = rect_se(5)
    cv_kernel = np.ones((5, 5), dtype=np.uint8)
    ours_dilate = dilate(binary, se)
    ours_erode = erode(binary, se)
    cv_dilate = cv2.dilate(
        binary.astype(np.uint8), cv_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    cv_erode = cv2.erode(
        binary.astype(np.uint8), cv_kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1
    ).astype(bool)
    dilate_difference = int(np.count_nonzero(ours_dilate != cv_dilate))
    erode_difference = int(np.count_nonzero(ours_erode != cv_erode))

    ccl_input = np.zeros((30, 35), dtype=np.uint8)
    ccl_input[2:8, 2:8] = 1
    ccl_input[12:20, 18:27] = 1
    ccl_input[24, 5] = 1
    ccl_input[25, 6] = 1
    ours_components = component_count(connected_components_labeling(ccl_input.astype(bool)))
    cv_count, _ = cv2.connectedComponents(ccl_input, connectivity=8)
    cv_components = cv_count - 1

    print(f"gaussian_vs_scipy_max_abs_error={scipy_error:.6g}")
    print(f"gaussian_vs_opencv_max_abs_error={cv_error:.6g}")
    print(f"otsu_ours={ours_otsu} otsu_opencv={int(cv_otsu)} difference={otsu_difference}")
    print(f"dilate_pixel_difference={dilate_difference}")
    print(f"erode_pixel_difference={erode_difference}")
    print(f"ccl_ours={ours_components} ccl_opencv={cv_components}")

    # SciPy uses symmetric reflection, while the project explicitly replicates edge pixels.
    interior = kernel.shape[0] // 2
    np.testing.assert_allclose(
        ours_gaussian[interior:-interior, interior:-interior],
        scipy_gaussian[interior:-interior, interior:-interior],
        atol=1e-6,
    )
    assert cv_error < 1e-6
    assert otsu_difference <= 1
    assert dilate_difference == 0
    assert erode_difference == 0
    assert ours_components == cv_components


def test_downscale_odd_shape_discards_only_incomplete_edge_blocks():
    image = np.arange(35, dtype=np.float64).reshape(5, 7)
    result = downscale_mean(image, 3)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[8.0, 11.0]])


@pytest.mark.parametrize("shape", [(1, 1), (9, 11), (11, 9)])
@pytest.mark.parametrize("value", [0, 255])
def test_pipeline_uniform_and_small_images_do_not_crash(shape, value):
    image = np.full((*shape, 3), value, dtype=np.uint8)
    boxes, intermediates = segment_page(image, use_blur=False, use_closing=False)
    assert isinstance(boxes, list)
    assert intermediates["gray"].ndim == 2


def test_pipeline_accepts_single_channel_odd_image():
    image = np.tile(np.arange(13, dtype=np.uint8), (15, 1)) * 10
    boxes, intermediates = segment_page(image, max_side=7, use_closing=False)
    assert isinstance(boxes, list)
    assert intermediates["gray"].ndim == 2
    assert intermediates["scale"] == 3


def test_process_rejects_imread_none_with_clear_value_error(tmp_path):
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(ValueError, match="(?i)(read|image|anh|file|path)"):
        process(str(missing), str(tmp_path / "output"))


def test_process_removes_stale_generated_panels(tmp_path):
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    image[8:56, 8:11] = 0
    image[8:56, 53:56] = 0
    image[8:11, 8:56] = 0
    image[53:56, 8:56] = 0
    image_path = tmp_path / "page.png"
    assert cv2.imwrite(str(image_path), image)
    panels_dir = tmp_path / "output" / "page" / "panels"
    panels_dir.mkdir(parents=True)
    stale = panels_dir / "panel_99.png"
    assert cv2.imwrite(str(stale), image[:1, :1])

    _, intermediates = segment_page(image)
    assert intermediates["diagonal_gutters"] == []
    boxes, _ = process(str(image_path), str(tmp_path / "output"))
    generated = sorted(panels_dir.glob("panel_*.png"))

    assert not stale.exists()
    assert len(generated) == len(boxes)
