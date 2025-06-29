from player_tracker import PlayerReIDTracker

if __name__ == "__main__":
    tracker = PlayerReIDTracker(
        det_model_path="models/best.pt",
        osnet_model_name="osnet_ain_x1_0"
    )
    tracker.track_video(
        video_path="assets/15sec_input_720p.mp4",
        output_path="outputs/reid_output.mp4",
        save_txt_path="outputs/reid_track.txt"
    )
    print("Tracking complete! Results saved to outputs/reid_output.mp4 and outputs/reid_track.txt")
