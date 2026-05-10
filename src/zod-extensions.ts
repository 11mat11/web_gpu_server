/* eslint-disable */
import { ZodType } from 'zod';

declare module 'zod' {
	interface ZodType<
		out Output = unknown,
		out Input = unknown,
		out Internals extends import('zod').core.$ZodTypeInternals<Output, Input> =
			import('zod').core.$ZodTypeInternals<Output, Input>,
	> {
		example(value: unknown): this;
	}
}

if (!(ZodType as any).prototype.example) {
	(ZodType as any).prototype.example = function example(this: any, value: unknown) {
		if (typeof this.meta === 'function') {
			const current = this.meta() ?? {};
			return this.meta({ ...current, example: value });
		}
		return this;
	};
}
