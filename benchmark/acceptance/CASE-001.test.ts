import { mount } from '@vue/test-utils'
import UserProfileForm from '@/components/UserProfileForm.vue'

describe('CASE-001 hidden acceptance', () => {
  it(
    'accepts exactly 30 characters and rejects names longer than 30',
    async () => {
      const wrapper = mount(
        UserProfileForm,
        {
          props: {
            initialName: 'A',
            initialEmail: 'a@example.com',
          },
        },
      )

      const nameInput = wrapper.get(
        'input[aria-label="Name"]',
      )

      const submitButton = wrapper.get(
        'button[type="submit"]',
      )

      // =====================================
      // 30 个字符：
      //
      // 必须是合法输入。
      // =====================================

      const validName = 'A'.repeat(30)

      await nameInput.setValue(
        validName,
      )

      expect(
        submitButton.attributes(
          'disabled',
        ),
      ).toBeUndefined()

      await wrapper
        .get('form')
        .trigger('submit')

      expect(
        wrapper.emitted('submit')?.[0],
      ).toEqual([
        {
          name: validName,
          email: 'a@example.com',
        },
      ])

      // =====================================
      // 31 个字符：
      //
      // 必须被现有 validation
      // 机制拒绝。
      // =====================================

      const invalidName = 'A'.repeat(31)

      await nameInput.setValue(
        invalidName,
      )

      expect(
        submitButton.attributes(
          'disabled',
        ),
      ).toBeDefined()

      // 不强绑定完整错误文案，
      // 但应该明确反映新的 30 字符限制。
      expect(
        wrapper.text(),
      ).toContain('30')
    },
  )
})