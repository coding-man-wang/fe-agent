import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { vi } from 'vitest'
import UsersView from '@/views/UsersView.vue'
import { useUsersStore } from '@/stores/users'

async function findAndSetPageSize(wrapper: ReturnType<typeof mount>, value: '10' | '20') {
  const select = wrapper
    .findAll('select')
    .find((item) => {
      const values = item.findAll('option').map((option) => (option.element as HTMLOptionElement).value)
      return values.includes('10') && values.includes('20')
    })

  if (select) {
    await select.setValue(value)
    return
  }

  const button = wrapper
    .findAll('button')
    .find((item) => new RegExp(`\\b${value}\\b`).test(item.text()))

  if (button) {
    await button.trigger('click')
    return
  }

  const radio = wrapper
    .findAll('input[type="radio"]')
    .find((item) => item.attributes('value') === value)

  if (radio) {
    await radio.setValue()
    return
  }

  throw new Error('No 10/20 page-size control found')
}

describe('CASE-006 hidden acceptance', () => {
  it('supports 10/20 page size while preserving filters and resetting page', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useUsersStore()
    const fetchSpy = vi.spyOn(store, 'fetchUsers').mockResolvedValue(undefined)

    const wrapper = mount(UsersView, {
      global: {
        plugins: [pinia],
        stubs: {
          UserFilters: true,
          UserTable: true,
          PaginationBar: true,
        },
      },
    })
    await nextTick()
    fetchSpy.mockClear()

    expect(store.pageSize).toBe(10)
    store.page = 2
    store.total = 37
    store.keyword = 'User 2'
    store.status = 'active'

    await findAndSetPageSize(wrapper, '20')
    await nextTick()
    expect(store.pageSize).toBe(20)
    expect(store.page).toBe(1)
    expect(store.keyword).toBe('User 2')
    expect(store.status).toBe('active')
    expect(store.totalPages).toBe(2)
    expect(fetchSpy).toHaveBeenCalledTimes(1)

    store.page = 2
    await findAndSetPageSize(wrapper, '10')
    await nextTick()
    expect(store.pageSize).toBe(10)
    expect(store.page).toBe(1)
    expect(store.totalPages).toBe(4)
    expect(fetchSpy).toHaveBeenCalledTimes(2)
  })
})
