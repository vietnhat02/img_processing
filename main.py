from utils import parse_arguments, load_config, get_screen_resolution, put_text
from processing import denoise, threshold, make_frame_show, create_final_frame
from trackers import count_square_circle
from logging_config import setup_logging
import cv2
import logging
import time
from motpy import ModelPreset, MultiObjectTracker

# Thiết lập logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting main function")
    args = parse_arguments()
    logger.debug(f"Arguments parsed successfully: {args}")

    # Load config
    logger.info("Loading configuration file")
    config = load_config("config.yaml")

    # Update config with command-line arguments
    logger.debug("Updating configuration with command-line arguments")
    if args.video_path:
        config['load_video'] = config.get('load_video', {})
        config['load_video']['path'] = args.video_path
    if args.name:
        config['name'] = args.name
    if args.mssv:
        config['mssv'] = args.mssv
    if args.denoise_c:
        config['blur'] = config.get('blur', {})
        config['blur']['denoise_kernel_size'] = args.denoise_c
    if args.low_threshold:
        config['threshold'] = config.get('threshold', {})
        config['threshold']['low'] = args.low_threshold
    if args.iterations_open:
        config['threshold'] = config.get('threshold', {})
        config['threshold']['iterations_open'] = args.iterations_open
    if args.iterations_close:
        config['threshold'] = config.get('threshold', {})
        config['threshold']['iterations_close'] = args.iterations_close

    # Open video file
    video_path = config.get("load_video", {}).get('path')
    logger.info(f"Opening video file at path: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Could not open video at path: {video_path}")
        return

    # Get original video properties
    frame_size_individual = (config.get('show_frame', {}).get('width', 640),
                             config.get('show_frame', {}).get('height', 480))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if unable to get FPS
    logger.info(f"Video properties - Frame size: {frame_size_individual}, FPS: {fps}")

    # Get screen resolution
    screen_width, screen_height = get_screen_resolution()
    logger.debug(f"Screen resolution: width={screen_width}, height={screen_height}")

    # Calculate optimal frame size for display (2x2 grid)
    # Ensure that the combined frame fits within the screen resolution with some margin
    margin = 100  # Pixels
    combined_width = min(frame_size_individual[0] * 2, screen_width - margin)
    combined_height = min(frame_size_individual[1] * 2, screen_height - margin)
    frame_size_combined = (combined_width, combined_height)
    logger.info(f"Combined frame size calculated: {frame_size_combined}")

    # Initialize VideoWriter with MP4V codec
    output_path = config.get('save_frame', {}).get('out_path', 'output.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec if necessary
    logger.info(f"Initializing VideoWriter with output path: {output_path}")

    try:
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size_combined
        )
        cv2.namedWindow('Video Processing Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Processing Result', frame_size_combined[0], frame_size_combined[1])

        # Initialize trackers outside the loop to maintain state across frames
        model_spec = ModelPreset.constant_acceleration_and_static_box_size_2d.value
        logger.info("Initializing trackers")
        tracker_square = MultiObjectTracker(
            dt=1/fps,
            model_spec=model_spec,
            active_tracks_kwargs={'min_steps_alive': 5, 'max_staleness': 15},
            tracker_kwargs={'max_staleness': 12}
        )
        tracker_circle = MultiObjectTracker(
            dt=1/fps,
            model_spec=model_spec,
            active_tracks_kwargs={'min_steps_alive': 5, 'max_staleness': 15},
            tracker_kwargs={'max_staleness': 12}
        )

        # Initialize total counts and counted IDs
        unique_square_ids = set()
        unique_circle_ids = set()
        counted_square_ids = set()  # Set to track counted square IDs
        counted_circle_ids = set()  # Set to track counted circle IDs

        logger.info("Starting video processing loop")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            try:
                # Start timing the processing of this frame
                frame_start_time = time.time()
                logger.debug("Processing new frame")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred_frame = denoise(gray, config.get("blur", {}))
                logger.debug("Frame denoised")

                ret_thresh, thresh_frame, pret_thresh_frame, contours = threshold(blurred_frame, config.get("threshold", {}))
                logger.debug(f"Thresholding completed, found {len(contours)} contours")

                processed_frame = count_square_circle(
                    thresh_frame,
                    frame.copy(),
                    config.get("counter", {}),
                    contours,
                    tracker_square,
                    tracker_circle,
                    unique_square_ids,
                    unique_circle_ids,
                    counted_square_ids,  # Pass the counted sets
                    counted_circle_ids
                )
                logger.debug("Counted squares and circles in frame")

                concatenated_frame = make_frame_show(
                    [frame, blurred_frame, pret_thresh_frame, processed_frame],
                    frame_size_individual
                )

                # Add text and overlays
                # Update config with current counts for display
                config['square_count'] = len(unique_square_ids)
                config['circle_count'] = len(unique_circle_ids)
                put_text(concatenated_frame, config)
                logger.debug("Overlay text added to frame")

                # Resize the concatenated frame to fit the combined frame size
                resized_concatenated_frame = cv2.resize(concatenated_frame, frame_size_combined)

                # Write the combined frame to the output video
                out.write(resized_concatenated_frame)
                logger.debug("Frame written to output video")

                # Display the combined frame
                cv2.imshow('Video Processing Result', resized_concatenated_frame)

                # Log the time taken to process this frame
                frame_end_time = time.time()
                logger.debug(f"Processed frame in {frame_end_time - frame_start_time:.2f} seconds")

            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                logger.debug("Traceback:", exc_info=True)
                continue

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break

        # After processing all frames or exiting early, display final counts
        logger.info("Final Counts:")
        logger.info(f"Total Squares: {len(unique_square_ids)}")
        logger.info(f"Total Circles: {len(unique_circle_ids)}")
        # Create a final frame displaying the total counts
        final_frame = create_final_frame(len(unique_square_ids), len(unique_circle_ids), frame_size_individual)

        # Add text to the final frame
        # Reset config counts for the final display
        final_config = {
            'square_count': len(unique_square_ids),
            'circle_count': len(unique_circle_ids),
            'mssv': config.get('mssv', 'MSSV'),
            'name': config.get('name', 'Name')
        }
        put_text(final_frame, final_config)
        logger.debug("Final frame created with total counts")

        # If you need to resize the final frame to match the output video size
        resized_final_frame = cv2.resize(final_frame, frame_size_individual)

        # Write the final frame 30 times to the output video
        logger.info("Writing final frame to output video 30 times")
        for _ in range(30):
            out.write(resized_final_frame)

        logger.info("Final frame written to output video")

        cv2.imshow('Video Processing Result', resized_final_frame)
        cv2.waitKey(0)

    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        logger.debug("Traceback:", exc_info=True)

    finally:
        # Release resources
        logger.info("Cleaning up resources")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
