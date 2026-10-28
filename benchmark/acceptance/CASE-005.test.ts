import {
  flushPromises,
  mount,
} from '@vue/test-utils'

import { nextTick } from 'vue'

import {
  afterEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest'


const EXPECTED_STATUSES = [
  'all',
  'pending',
  'paid',
  'cancelled',
]


function findStatusSelect(
  wrapper: ReturnType<typeof mount>,
) {
  return wrapper
    .findAll('select')
    .find((candidate) => {
      const values = candidate
        .findAll('option')
        .map(
          (option) => (
            option.element as HTMLOptionElement
          ).value,
        )

      return EXPECTED_STATUSES.every(
        (status) =>
          values.includes(status),
      )
    })
}


describe(
  'CASE-005 hidden acceptance',
  () => {
    afterEach(() => {
      vi.useRealTimers()
      vi.unstubAllEnvs()
      vi.resetModules()
      vi.clearAllMocks()
      vi.doUnmock(
        '@/api/requestClient',
      )
    })

    it(
      'filters orders consistently in mock mode',
      async () => {
        // ==================================
        // Mock Mode
        // ==================================

        vi.useFakeTimers()

        vi.stubEnv(
          'VITE_API_MODE',
          'mock',
        )

        vi.resetModules()

        const [
          { default: OrdersView },
          { orders },
        ] = await Promise.all([
          import(
            '@/views/OrdersView.vue'
          ),
          import(
            '@/mocks/database'
          ),
        ])

        const wrapper = mount(
          OrdersView,
        )

        await vi.runAllTimersAsync()
        await flushPromises()
        await nextTick()

        // ==================================
        // Status options
        // ==================================

        const select =
          findStatusSelect(
            wrapper,
          )

        if (!select) {
          throw new Error(
            'Order status select not found',
          )
        }

        const optionValues = select
          .findAll('option')
          .map(
            (option) => (
              option.element as HTMLOptionElement
            ).value,
          )

        expect(
          optionValues,
        ).toEqual(
          expect.arrayContaining(
            EXPECTED_STATUSES,
          ),
        )

        // ==================================
        // Initial "all"
        // ==================================

        expect(
          wrapper.findAll(
            'tbody tr',
          ).length,
        ).toBe(
          orders.length,
        )

        // ==================================
        // pending / paid / cancelled
        //
        // 切换状态必须重新走 Mock API，
        // 最终页面只展示对应状态。
        // ==================================

        for (
          const status
          of [
            'pending',
            'paid',
            'cancelled',
          ]
        ) {
          await select.setValue(
            status,
          )

          await vi
            .runAllTimersAsync()

          await flushPromises()
          await nextTick()

          const rows =
            wrapper.findAll(
              'tbody tr',
            )

          expect(
            rows.length,
          ).toBeGreaterThan(0)

          expect(
            rows.every(
              (row) => {
                const cells =
                  row.findAll(
                    'td',
                  )

                return (
                  cells[3]
                  ?.text()
                  .toLowerCase()
                  === status
                )
              },
            ),
          ).toBe(true)
        }

        // ==================================
        // all restores all orders
        // ==================================

        await select.setValue(
          'all',
        )

        await vi.runAllTimersAsync()
        await flushPromises()
        await nextTick()

        expect(
          wrapper.findAll(
            'tbody tr',
          ).length,
        ).toBe(
          orders.length,
        )
      },
    )

    it(
      'passes selected status to the remote orders request',
      async () => {
        // ==================================
        // Remote Mode
        //
        // 不请求真实网络。
        // 只检查 orders API 最终给
        // requestClient 的 config。
        // ==================================

        vi.stubEnv(
          'VITE_API_MODE',
          'remote',
        )

        const request = vi
          .fn()
          .mockResolvedValue([])

        vi.doMock(
          '@/api/requestClient',
          () => ({
            request,
          }),
        )

        vi.resetModules()

        const {
          default: OrdersView,
        } = await import(
          '@/views/OrdersView.vue'
        )

        const wrapper = mount(
          OrdersView,
        )

        await flushPromises()
        await nextTick()

        const select =
          findStatusSelect(
            wrapper,
          )

        if (!select) {
          throw new Error(
            'Order status select not found',
          )
        }

        const initialCallCount =
          request.mock.calls.length

        await select.setValue(
          'paid',
        )

        await flushPromises()
        await nextTick()

        // 切换筛选条件后，
        // 必须重新请求。
        expect(
          request.mock.calls.length,
        ).toBeGreaterThan(
          initialCallCount,
        )

        // Remote API 必须把 status
        // 通过 request params 传递。
        expect(
          request,
        ).toHaveBeenCalledWith(
          expect.objectContaining({
            method: 'GET',
            url: '/orders',
            params:
              expect.objectContaining({
                status: 'paid',
              }),
          }),
        )
      },
    )
  },
)