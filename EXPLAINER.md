# Interview explainer: wideband signal recognition

## What problem does this solve?

A receiver can capture a wide chunk of radio spectrum as complex I/Q samples. The system should point out where signals appear in time and frequency, name the broad modulation family, and inspect PAM-like signals more closely. An operator can use these boxes and labels to monitor a band or investigate interference.

## Why use an image detector?

A spectrogram turns time into the horizontal direction and frequency into the vertical direction. A signal then looks like a patch or stripe. An object detector can locate several such patches and assign a broad class in one pass. The energy baseline only groups bright connected regions; it gives a geometry comparison but cannot decide a class.

## Why does the detector use one PAM class?

PSK and QAM differ in the symbols carried by the waveform. Those symbols are not directly visible in a power-only spectrogram. After normalization, their patches can have very similar shape and brightness, so the detector merges them into PAM. The second classifier uses the complex I and Q values, where phase and amplitude symbol patterns remain available.

## What does the spectrogram resolution mean?

The analysis window spans 512 input samples and advances without overlap. That gives a time step of 512 samples. A 4096-point FFT gives finely spaced plotted frequency bins by zero-padding, but the true ability to distinguish nearby tones is still limited mainly by the 512-sample window. A shorter signal may occupy only a few columns. The approved minimum duration and bandwidth were chosen so the smallest synthetic box covers at least five resized pixels on each image axis; the notebook repeats the low-SNR visibility experiment before training.

The image stretches the time and frequency axes to a square detector input. This preserves the box mapping only because labels are converted using the same covered-sample and pixel geometry. Labels come from the signal generator parameters, never from looking at the rendered image.

## What does a box mean physically?

Its left and right edges are the signal start and end in samples. Its bottom and top edges are the low and high normalized frequencies in cycles per sample. With a real sample rate, normalized frequency can later be converted to hertz. The final STFT window does not cover the last 64 samples of a one-million-sample segment, so the box-to-image map uses the samples actually covered by the windows.

## Why filter and downconvert before PAM classification?

The detector supplies an estimated frequency band. The extraction block shifts that band's center to zero frequency, applies a linear-phase low-pass filter, compensates for the filter delay, and cuts frames from inside the estimated time interval. This makes the classifier focus on the modulation symbols instead of the full wideband recording.

If the estimated center is wrong, residual carrier rotation remains in the frame. The agreed edge jitter is five percent of box bandwidth per frequency edge; when both edges shift together, the largest center error is five percent of the box bandwidth. At the shifted-profile maximum bandwidth of 0.12 cycles per sample, that is 0.006 cycles per sample, or about 6.14 phase cycles in a 1024-sample frame. The filter uses an odd 1111-tap Kaiser-windowed FIR at minimum bandwidth, with 555 samples of group delay. Unit tests check its passband ripple and stopband attenuation.

Too narrow a filter clips wanted symbols and spreads their pulse shape. Too wide a filter admits nearby signals and noise. The generator therefore enforces a guard between simultaneous signals; predictions closer than the guard are marked unresolved rather than treated as clean isolated signals.

## What should happen as SNR falls?

The spectrogram becomes less distinct and the I/Q symbols become less reliable. Detection recall, box overlap, and PAM accuracy should generally decline as noise rises. An unexpected large improvement at lower SNR, or a sharp improvement only on test data, would make me inspect the SNR calculation, split, normalization, and label alignment for a bug.

Global SNR is computed from total clean signal power divided by total added noise power across the full segment. A PAM signal's approximate in-band SNR is also saved per signal. These differ because several signals share a segment and because the noise integrated over one signal's bandwidth is different from total wideband noise.

## How do we avoid optimistic testing?

Parent recording IDs are assigned to train, validation, or test before any detector training. Every segment under a parent ID stays in one split. The additional shifted test set uses different bandwidth and SNR ranges. Bootstrap intervals sample parent recordings, not individual detections or frames. Any low-support class or SNR cell is explicitly marked.

The current synthetic implementation uses independent seeded segment realizations grouped under parent IDs for split and bootstrap bookkeeping. That is a scaled test protocol, not a claim that a continuous real recording was captured.

## What does the PAM CNN learn?

The classifier receives short complex I/Q frames, represented as two channels: I and Q. A small one-dimensional CNN learns local waveform patterns. It predicts one of the PAM subtypes or Others. Others is trained from the agreed AM-DSB, FM, two-FSK, and OOK signals using the same extraction steps. OFDM is excluded from this classifier's Others training set by design.

Accuracy with generator-defined boxes measures the classifier when extraction is ideal. Accuracy with detector boxes measures the effect of real localization errors. The joint end-to-end score is stricter: a detection counts only when its box overlaps enough, its detector class is correct, and its PAM subtype is correct when applicable.

## Why keep the energy baseline?

A simple connected-component detector creates candidate boxes from bright spectrogram regions. We calibrate its confidence on validation data to the same 80 percent precision operating target as YOLO when possible. Then we compare class-agnostic recall and mean IoU on the untouched test recordings. YOLO class mAP is reported separately because the energy baseline has no class labels.

If YOLO does not improve geometry enough to justify its complexity, the result is still useful: it shows the limit of a simple brightness-based baseline on this synthetic layout. No score is copied from the paper.

## What do we deliberately leave out?

The prototype does not model fading, frequency offset beyond deliberate extraction-error tests, oscillator phase noise, IQ imbalance, DC offset, nonlinear hardware distortion, ADC quantization, AGC behavior, or sample-clock drift. It also does not validate the synthetic waveforms as standards-compliant emitters, cover every paper class, or establish real-world receiver performance. OOK versus narrow AM-DSB and FM versus PAM confusion are known cases to measure. External real data is deferred until access and license are checked.
