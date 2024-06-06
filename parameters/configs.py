class Config:
    def __init__(self, sample_rate, min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft, dim_square_spec):
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.frame_length = frame_length
        self.hop_length_frame = hop_length_frame
        self.hop_length_frame_noise = hop_length_frame_noise
        self.nb_samples = nb_samples
        self.n_fft = n_fft
        self.hop_length_fft = hop_length_fft
        self.dim_square_spec = dim_square_spec

def parameters(config):
    return config