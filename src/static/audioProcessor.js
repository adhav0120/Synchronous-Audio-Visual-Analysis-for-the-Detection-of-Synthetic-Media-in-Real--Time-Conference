// This script runs in a separate thread to process audio efficiently.
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096; // Buffer size before sending data
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const inputChannel = input[0];

        if (inputChannel) {
            // Add incoming audio data to our buffer
            for (let i = 0; i < inputChannel.length; i++) {
                this.buffer[this.bufferIndex++] = inputChannel[i];
                
                // When the buffer is full, send it to the main thread
                if (this.bufferIndex === this.bufferSize) {
                    this.port.postMessage(this.buffer);
                    this.bufferIndex = 0; // Reset buffer
                }
            }
        }
        
        // Keep the processor running
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
