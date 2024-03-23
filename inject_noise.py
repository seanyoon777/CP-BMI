import numpy as np 

def add_gaussian_noise(data, strength):
    mu = 0
    var = strength
    sigma = var**0.5
    noise = np.random.normal(mu, sigma, data.shape)
    return data + noise

def add_impulse_noise(data, strength):
    t_plus = .5
    noise_t = 10
    new_data = np.copy(data)
    n_timestamps = new_data.shape[-1]
    for sample_num in range(new_data.shape[0]):
        # Determine the number of impulses based on the probability
        n_impulses = int(np.ceil(n_timestamps * strength) / 5)
        impulse_timestamps = np.random.choice(int(n_timestamps-noise_t), n_impulses, replace=False)
        # Apply impulses
        for timestamp in impulse_timestamps:
            if np.random.rand() > t_plus: 
                new_data[sample_num, 0, :, timestamp:(timestamp + noise_t)] += 5
            else:
                new_data[sample_num, 0, :, timestamp:(timestamp + noise_t)] -= 5
    return new_data

def add_motion_artifacts(data, strength):
    """
    Simulate a motion artifact in EEG data.

    Parameters:
    - eeg_data: numpy array of shape (n_samples, 1, n_channels, timestamps)
    - lm: The length of the motion artifact affecting consecutive timestamps.

    Returns:
    - Modified EEG data with simulated motion artifact.
    """
    eeg_data = data.copy()
    n_samples, _, n_channels, n_timestamps = eeg_data.shape
    
    # Choose a random start point for the artifact in each sample
    for sample_idx in range(n_samples):
        duration = 250
        start_time = np.random.randint(0, n_timestamps - duration)
        end_time = start_time + duration
        
        # Generate a more varied artifact shape
        artifact = np.sin(np.linspace(-np.pi / 2, np.pi / 2, duration)) * np.max(np.abs(eeg_data)) * strength * 5 # Sinusoidal artifact
        
        for channel_idx in range(n_channels):
            eeg_data[sample_idx, 0, channel_idx, start_time:end_time] += artifact

    return eeg_data