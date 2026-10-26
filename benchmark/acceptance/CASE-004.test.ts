import { mount } from '@vue/test-utils'

import UserFilters from '@/components/UserFilters.vue'
import { users } from '@/mocks/database'
import { mockListUsers } from '@/mocks/service'

import type {
  UserStatus,
} from '@/types/domain'


describe(
  'CASE-004 hidden acceptance',
  () => {
    it(
      'supports pending users without regressing existing status filters',
      async () => {
        // ==================================
        // 1. Compile-time contract
        //
        // 如果 UserStatus 没增加 pending，
        // typecheck / hidden test 编译会失败。
        // ==================================

        const pendingStatus: UserStatus =
          'pending'

        // ==================================
        // 2. UI Filter
        // ==================================

        const wrapper = mount(
          UserFilters,
          {
            props: {
              keyword: '',
              status: 'all',
            },
          },
        )

        const select = wrapper.get(
          'select[aria-label="User status"]',
        )

        const options = select
          .findAll('option')
          .map((option) => ({
            value: (
              option.element
              as HTMLOptionElement
            ).value,
            text: option.text(),
          }))

        expect(
          options.map(
            (option) => option.value,
          ),
        ).toEqual(
          expect.arrayContaining([
            'all',
            'active',
            'disabled',
            'pending',
          ]),
        )

        expect(
          options.some(
            (option) =>
              option.value ===
                'pending'
              &&
              option.text ===
                'Pending',
          ),
        ).toBe(true)

        // ==================================
        // 3. Mock database
        // ==================================

        const pendingUsers =
          users.filter(
            (user) =>
              user.status ===
              pendingStatus,
          )

        expect(
          pendingUsers.length,
        ).toBeGreaterThan(0)

        // ==================================
        // 4. Existing generic filtering
        //
        // 不要求修改 service.ts。
        // 现有 status equality filter
        // 应天然处理 pending。
        // ==================================

        const pendingResult =
          await mockListUsers({
            keyword: '',
            status: pendingStatus,
            page: 1,
            pageSize: 100,
          })

        expect(
          pendingResult.total,
        ).toBe(
          pendingUsers.length,
        )

        expect(
          pendingResult.items.length,
        ).toBeGreaterThan(0)

        expect(
          pendingResult.items.every(
            (user) =>
              user.status ===
              'pending',
          ),
        ).toBe(true)

        // ==================================
        // 5. Existing statuses preserved
        // ==================================

        const activeResult =
          await mockListUsers({
            keyword: '',
            status: 'active',
            page: 1,
            pageSize: 100,
          })

        expect(
          activeResult.items.every(
            (user) =>
              user.status ===
              'active',
          ),
        ).toBe(true)

        const disabledResult =
          await mockListUsers({
            keyword: '',
            status: 'disabled',
            page: 1,
            pageSize: 100,
          })

        expect(
          disabledResult.items.every(
            (user) =>
              user.status ===
              'disabled',
          ),
        ).toBe(true)

        const allResult =
          await mockListUsers({
            keyword: '',
            status: 'all',
            page: 1,
            pageSize: 100,
          })

        expect(
          allResult.total,
        ).toBe(
          users.length,
        )
      },
    )
  },
)