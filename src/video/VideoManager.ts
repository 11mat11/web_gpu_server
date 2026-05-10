import { readdir, readFile } from 'node:fs/promises';
import { resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const VIDEO_FILE_SUFFIX = '.bin';

export class VideoManager {
	private static instance: VideoManager | null = null;

	private readonly videoDir: string;

	private constructor() {
		const currentDir = fileURLToPath(new URL('.', import.meta.url));
		this.videoDir = resolve(currentDir);
	}

	static getInstance(): VideoManager {
		if (!VideoManager.instance) {
			VideoManager.instance = new VideoManager();
		}
		return VideoManager.instance;
	}

	async listVideos(): Promise<string[]> {
		const entries = await readdir(this.videoDir, { withFileTypes: true });
		return entries
			.filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith(VIDEO_FILE_SUFFIX))
			.map((entry) => entry.name)
			.sort((a, b) => a.localeCompare(b));
	}

	async getVideoBuffer(fileName: string): Promise<Buffer> {
		if (!fileName.toLowerCase().endsWith(VIDEO_FILE_SUFFIX)) {
			throw new Error('Only .bin video files are supported.');
		}

		// Protect against path traversal: only allow simple file names from src/video.
		const safeName = basename(fileName);
		if (safeName !== fileName) {
			throw new Error('Invalid file name.');
		}

		return readFile(resolve(this.videoDir, safeName));
	}
}
