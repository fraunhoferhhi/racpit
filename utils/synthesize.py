from utils import radar, skeletons as sk
import numpy as np
# import scipy.io


def synthetic_radar(skeletons, config, frame_times=None):
    """
    :param skeletons: skeleton data as provided by the Daq module
    :param config: the configuration of the radar
    :param frame_times: the start times of all frames in the format [f1,f2,...]
    :return: Synthetic data in the format (nFrames, 1, nChirps, nSamples)
    """

    radarLoc = [p[0] for p in config["position"]]
    timestamps = skeletons.index
    startTime = timestamps[0].total_seconds()
    endTime = timestamps[-1].total_seconds()

    name = config["cfg"]
    f_low = config["LowerFrequency"] * 1e6
    f_up = config["UpperFrequency"] * 1e6
    numChirps = config["ChirpsPerFrame"]
    numSamples = config["SamplesPerChirp"]
    framePeriod = config["FramePeriod"] * 1e-6
    chirpPeriod = config["ChirpToChirpTime"] * 1e-9
    samplePeriod = 1e-3 / config["AdcSamplerate"]

    print("Create synthetic data for configuration {}".format(name))
    if frame_times is None:
        frame_times = np.arange(startTime, endTime, framePeriod)
        print("Creating {} frames of synthetic data with framePeriod {}".format(len(frame_times), framePeriod))
    else:
        print("Creating {} frames of synthetic data from provided frameTimes".format(len(frame_times)))

    C = 299792458  # m / s

    numFrames = len(frame_times)

    # array containing the start time of all chirps in the format [frames, chirps]
    chirpTimes = np.linspace(frame_times, frame_times + (numChirps - 1) * chirpPeriod, numChirps).T

    # Interpolation to increase temporal resolution of the skeleton data
    # returns segment position at each chirp start time
    sk_data = sk.interpolate(skeletons, chirpTimes.flatten())

    # if saveMatlab:
    #     print("Save skeletons for matlab to location: {}".format(saveMatlab.absolute()))
    #     scipy.io.savemat(saveMatlab, {'data': data,
    #                                   'timestamp': timestamps.to_numpy()})

    # Height of the person
    heightNose = np.mean([sk_keypoints[2, 0] for sk_keypoints in skeletons.Data])
    Height = heightNose + 0.16
    # body segments length (meter)
    headlen = 0.130 * Height
    shoulderlen = (0.259 / 2) * Height
    torsolen = 0.288 * Height
    hiplen = (0.191 / 2) * Height
    upperleglen = 0.245 * Height
    # lowerleglen = 0.246 * Height
    # footlen = 0.143 * Height
    upperarmlen = 0.188 * Height
    lowerarmlen = 0.152 * Height
    # Ht = upperleglen + lowerleglen

    # Get coordinates of the person
    head = (sk_data[:, :, 3] + sk_data[:, :, 4]) / 2

    neck = head.copy()
    neck[:, 2] -= 0.17

    base = (sk_data[:, :, 11] + sk_data[:, :, 12]) / 2
    base[:, 2] += 0.1

    lshoulder = sk_data[:, :, 5]
    lelbow = sk_data[:, :, 7]
    lhand = sk_data[:, :, 9]

    lhip = sk_data[:, :, 11]
    lknee = sk_data[:, :, 13]
    lankle = sk_data[:, :, 15]
    # ltoe = lankle.copy()

    rshoulder = sk_data[:, :, 6]
    relbow = sk_data[:, :, 8]
    rhand = sk_data[:, :, 10]

    rhip = sk_data[:, :, 12]
    rknee = sk_data[:, :, 14]
    rankle = sk_data[:, :, 16]
    # rtoe = rankle.copy()

    torso = (neck + base) / 2
    lupperarm = (lshoulder + lelbow) / 2
    rupperarm = (rshoulder + relbow) / 2
    lupperleg = (lhip + lknee) / 2
    rupperleg = (rhip + rknee) / 2
    llowerleg = (lankle + lknee) / 2
    rlowerleg = (rankle + rknee) / 2

    # Based on "A global human walking model with real-time kinematic personification",
    # by R. Boulic, N.M. Thalmann, and D. Thalmann % The Visual Computer, vol .6, pp .344 - 358, 1990
    segments = [
        sk.Segment(
            segmentPositions=head,
            ellipsoidParams=[0.1, 0.1, headlen / 2],
            aspect=head - neck
        ),
        sk.Segment(
            segmentPositions=torso,
            ellipsoidParams=[0.15, 0.15, torsolen / 2],
            aspect=neck - base
        ),
        sk.Segment(
            segmentPositions=lshoulder,
            ellipsoidParams=[0.06, 0.06, shoulderlen / 2],
            aspect=lshoulder - neck
        ),
        sk.Segment(
            segmentPositions=rshoulder,
            ellipsoidParams=[0.06, 0.06, shoulderlen / 2],
            aspect=rshoulder - neck
        ),
        sk.Segment(
            segmentPositions=lupperarm,
            ellipsoidParams=[0.06, 0.06, upperarmlen / 2],
            aspect=lshoulder - lelbow
        ),
        sk.Segment(
            segmentPositions=rupperarm,
            ellipsoidParams=[0.06, 0.06, upperarmlen / 2],
            aspect=rshoulder - relbow
        ),
        sk.Segment(
            segmentPositions=lhand,
            ellipsoidParams=[0.05, 0.05, lowerarmlen / 2],
            aspect=lelbow - lhand
        ),
        sk.Segment(
            segmentPositions=rhand,
            ellipsoidParams=[0.05, 0.05, lowerarmlen / 2],
            aspect=relbow - rhand
        ),
        sk.Segment(
            segmentPositions=lhip,
            ellipsoidParams=[0.07, 0.07, hiplen / 2],
            aspect=lhip - base
        ),
        sk.Segment(
            segmentPositions=rhip,
            ellipsoidParams=[0.07, 0.07, hiplen / 2],
            aspect=rhip - base
        ),
        sk.Segment(
            segmentPositions=lupperleg,
            ellipsoidParams=[0.07, 0.07, upperleglen / 2],
            aspect=lknee - lhip
        ),
        sk.Segment(
            segmentPositions=rupperleg,
            ellipsoidParams=[0.07, 0.07, upperleglen / 2],
            aspect=rknee - rhip
        ),
        sk.Segment(
            segmentPositions=llowerleg,
            ellipsoidParams=[0.06, 0.06, upperleglen / 2],
            aspect=lankle - lknee
        ),
        sk.Segment(
            segmentPositions=rlowerleg,
            ellipsoidParams=[0.06, 0.06, upperleglen / 2],
            aspect=rankle - rknee
        )
    ]

    # define timestamps of a single chirp
    t = np.linspace(0, (numSamples - 1) * samplePeriod, numSamples)

    bw = f_up - f_low
    alpha = bw / t[-1]
    f_c = f_low + (f_up - f_low) / 2
    s = np.zeros((numSamples, np.size(chirpTimes)))

    dr, r_max = radar.range_axis(bw, numSamples)

    for segment in segments:
        r_total = segment.calculateRange(radarLoc)
        phi, theta = segment.calculateAngles(radarLoc)
        rcs = segment.calculateRCS(phi, theta)
        amp = np.sqrt(rcs)

        # Formally it should not be squared, but squaring resembles the effect of the low pass filter
        fspl = (4 * np.pi * r_total * f_c / C) ** 2

        # calculates the signal for all chirps c and samples s in the format [[chirp1s1, chirp2s1, ..., chirpns1]
        #                                                                    [chirp1s2, chirp2s2, ..., chripns2 ]
        #                                                                    [              ...                 ]
        #                                                                    [chrip1sm, chrip2sm, ..., chripnsm]]
        s_segment = (amp / fspl) * np.cos(2 * np.pi * (2 * f_low * r_total / C +
                                                       2 * alpha * np.outer(t, r_total) / C -
                                                       2 * alpha * r_total.reshape(1, -1) ** 2 / C ** 2))

        # print("amp shape: {}".format(amp.shape))
        # print("np.outer shape: {}".format(np.outer(t, r_total).shape))
        # print("r_total.reshape: {}".format(r_total.reshape(1,-1).shape))

        # set whole chirp to zero if segment is out of sight
        s_segment[:, r_total > r_max] = 0
        s += s_segment

    s = s.T.reshape(numFrames, numChirps, numSamples)
    s = np.expand_dims(s, axis=1)

    print("Synthetic data with shape {} successfully created".format(s.shape))
    return s
