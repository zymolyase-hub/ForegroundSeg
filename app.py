from foregroundseg.visualization import SegmentationVisualizer

visualizer = SegmentationVisualizer()

overlay = visualizer.overlay(image, mask)

foreground = visualizer.foreground(image, mask)

background = visualizer.background(image, mask)

mask_image = visualizer.mask_to_image(mask)
