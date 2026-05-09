import { ZodType } from 'zod'

declare module 'zod' {
  // Extend Zod schemas with OpenAPI-like example metadata.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  interface ZodType<
    out Output = unknown,
    out Input = unknown,
    out Internals extends import('zod').core.$ZodTypeInternals<Output, Input> = import('zod').core.$ZodTypeInternals<Output, Input>
  > {
    example(value: unknown): this
  }
}

// Attach a no-op example() helper that stores metadata on the schema definition.
// This keeps runtime behavior unchanged while enabling Swagger tooling.
if (!(ZodType as any).prototype.example) {
  ;(ZodType as any).prototype.example = function example(this: any, value: unknown) {
    if (typeof this.meta === 'function') {
      const current = this.meta() ?? {}
      return this.meta({ ...current, example: value })
    }
    return this
  }
}
