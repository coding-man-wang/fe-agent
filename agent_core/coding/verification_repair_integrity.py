from __future__ import annotations

import re

from collections import Counter
from dataclasses import dataclass

from agent_core.coding.patch_generation_model import (
    FilePatchProposal,
    ValidatedFilePatch,
)


class VerificationRepairIntegrityError(ValueError):
    """
    Verification Repair
    降低了原测试义务。
    """


@dataclass(frozen=True)
class VerificationRepairIntegrityViolation:
    """
    一条确定性的 Repair Integrity 违规。
    """

    code: str

    relative_path: str

    message: str


@dataclass(frozen=True)
class VerificationRepairIntegrityResult:
    """
    Repair Integrity 检查结果。
    """

    allowed: bool

    violations: tuple[
        VerificationRepairIntegrityViolation,
        ...,
    ]

    def format_message(
        self,
    ) -> str:

        if not self.violations:
            return "verification repair integrity passed"

        return "; ".join(
            (f"{item.code}: " f"{item.relative_path}: " f"{item.message}")
            for item in self.violations
        )


class VerificationRepairIntegrityGuard:
    """
    Verification Repair Integrity Guard。

    目标不是判断测试“写得好不好”。

    它只保护一个非常重要的不变量：

        Repair 可以修改
        How to prove

        但不能减少
        What must still be proven。

    v1 使用确定性静态规则防止：

    1. 新增 skip / todo / fixme / fail / only；
    2. 删除原有 Test Case；
    3. 减少 assertion；
    4. 改变 Repair Patch Scope；
    5. 改变 Patch Action。

    当前主要针对：

        Vitest
        Playwright
        Jest-style test syntax

    后续如果需要更高精度，
    再升级到 AST-based Test Integrity。
    """

    # ==================================
    # 禁止 Repair 新增的测试修饰符
    #
    # skip:
    #   直接跳过测试
    #
    # todo:
    #   把测试转成待办
    #
    # fixme:
    #   Playwright 可跳过/标记问题
    #
    # fail:
    #   Playwright 将失败转成 expected failure
    #
    # only:
    #   可能导致只运行单个测试/测试组
    # ==================================

    _DISABLING_MODIFIER_PATTERN = re.compile(
        r"""
        \b
        (?:
            test
            \s*\.\s*
            describe
            |
            test
            |
            it
            |
            describe
        )
        \s*\.\s*
        (?P<modifier>
            skip
            |
            todo
            |
            fixme
            |
            fail
            |
            only
        )
        \s*
        \(
        """,
        re.VERBOSE,
    )

    # ==================================
    # 普通 test / it
    #
    # 支持：
    #
    # test('name', ...)
    # test.only('name', ...)
    # test.skip('name', ...)
    # it('name', ...)
    #
    # describe 不属于 Test Case，
    # 所以不计入。
    # ==================================

    _SINGLE_QUOTE_TEST_PATTERN = re.compile(
        r"""
        \b
        (?:test|it)
        (?:
            \s*\.\s*
            (?:
                only
                |
                skip
                |
                todo
                |
                fixme
                |
                fail
                |
                concurrent
                |
                serial
            )
        )?
        \s*
        \(
        \s*
        '
        (?P<title>
            (?:\\.|[^'\\])*
        )
        '
        """,
        re.VERBOSE,
    )

    _DOUBLE_QUOTE_TEST_PATTERN = re.compile(
        r"""
        \b
        (?:test|it)
        (?:
            \s*\.\s*
            (?:
                only
                |
                skip
                |
                todo
                |
                fixme
                |
                fail
                |
                concurrent
                |
                serial
            )
        )?
        \s*
        \(
        \s*
        "
        (?P<title>
            (?:\\.|[^"\\])*
        )
        "
        """,
        re.VERBOSE,
    )

    _BACKTICK_TEST_PATTERN = re.compile(
        r"""
        \b
        (?:test|it)
        (?:
            \s*\.\s*
            (?:
                only
                |
                skip
                |
                todo
                |
                fixme
                |
                fail
                |
                concurrent
                |
                serial
            )
        )?
        \s*
        \(
        \s*
        `
        (?P<title>
            [^`]*
        )
        `
        """,
        re.VERBOSE,
    )

    # ==================================
    # Assertions
    #
    # Vitest / Jest / Playwright：
    #
    # expect(...)
    # expect.soft(...)
    # expect.poll(...)
    #
    # 同时兼容常见：
    #
    # assert(...)
    # assert.equal(...)
    # ==================================

    _EXPECT_PATTERN = re.compile(
        r"""
        \b
        expect
        (?:
            \s*\.\s*
            (?:
                soft
                |
                poll
            )
        )?
        \s*
        \(
        """,
        re.VERBOSE,
    )

    _ASSERT_PATTERN = re.compile(
        r"""
        \b
        assert
        (?:
            \s*\.\s*
            [A-Za-z_][A-Za-z0-9_]*
        )?
        \s*
        \(
        """,
        re.VERBOSE,
    )

    def evaluate(
        self,
        *,
        failed_proposal: FilePatchProposal,
        repaired_proposal: FilePatchProposal,
    ) -> VerificationRepairIntegrityResult:
        """
        对比：

            Verification 失败的 Proposal

                    ↓

            Verification Repair Proposal

        确认 Repair 没有降低测试义务。
        """

        violations: list[VerificationRepairIntegrityViolation] = []

        failed_map = {patch.relative_path: patch for patch in failed_proposal.patches}

        repaired_map = {
            patch.relative_path: patch for patch in repaired_proposal.patches
        }

        # ==================================
        # 1. Patch Scope 不能变化
        #
        # VerificationRepairService
        # 已经检查一次。
        #
        # Guard 再检查一次属于
        # Defense in Depth。
        # ==================================

        failed_paths = set(failed_map)

        repaired_paths = set(repaired_map)

        missing_paths = sorted(failed_paths - repaired_paths)

        extra_paths = sorted(repaired_paths - failed_paths)

        for path in missing_paths:
            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_target_removed"),
                    relative_path=path,
                    message=("Repair 删除了原 " "Patch Proposal Target"),
                )
            )

        for path in extra_paths:
            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_target_added"),
                    relative_path=path,
                    message=("Repair 新增了原 " "Patch Proposal " "之外的 Target"),
                )
            )

        # Scope 已经不一致时，
        # 只检查两边共同文件。
        common_paths = sorted(failed_paths & repaired_paths)

        for path in common_paths:

            failed_patch = failed_map[path]

            repaired_patch = repaired_map[path]

            violations.extend(
                self._evaluate_patch(
                    failed_patch=(failed_patch),
                    repaired_patch=(repaired_patch),
                )
            )

        return VerificationRepairIntegrityResult(
            allowed=(not violations),
            violations=tuple(violations),
        )

    def enforce(
        self,
        *,
        failed_proposal: FilePatchProposal,
        repaired_proposal: FilePatchProposal,
    ) -> None:
        """
        Harness Gate API。

        不通过直接抛异常。
        """

        result = self.evaluate(
            failed_proposal=(failed_proposal),
            repaired_proposal=(repaired_proposal),
        )

        if result.allowed:
            return

        raise (VerificationRepairIntegrityError(result.format_message()))

    def _evaluate_patch(
        self,
        *,
        failed_patch: ValidatedFilePatch,
        repaired_patch: ValidatedFilePatch,
    ) -> list[VerificationRepairIntegrityViolation]:

        violations: list[VerificationRepairIntegrityViolation] = []

        path = failed_patch.relative_path

        # ==================================
        # 2. Action 不能变化
        #
        # create 不能变 modify，
        # modify 不能变 create。
        # ==================================

        if failed_patch.action != repaired_patch.action:
            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_action_changed"),
                    relative_path=path,
                    message=(
                        "Repair 改变了 "
                        "Patch Action: "
                        f"{failed_patch.action}"
                        " -> "
                        f"{repaired_patch.action}"
                    ),
                )
            )

        failed_content = failed_patch.after_content

        repaired_content = repaired_patch.after_content

        # ==================================
        # 去除 JS/TS Comments。
        #
        # 避免：
        #
        # // 不要使用 test.skip(...)
        #
        # 被错误当作真正 skip。
        # ==================================

        failed_code = self._strip_js_comments(failed_content)

        repaired_code = self._strip_js_comments(repaired_content)

        # ==================================
        # 3. 不允许新增 Disable Modifier
        # ==================================

        before_modifiers = self._count_disabling_modifiers(failed_code)

        after_modifiers = self._count_disabling_modifiers(repaired_code)

        all_modifiers = set(before_modifiers) | set(after_modifiers)

        for modifier in sorted(all_modifiers):

            before_count = before_modifiers[modifier]

            after_count = after_modifiers[modifier]

            if after_count <= before_count:
                continue

            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_test_disabled"),
                    relative_path=path,
                    message=(
                        "Repair 新增了测试"
                        "降级修饰符 "
                        f"{modifier}: "
                        f"{before_count}"
                        " -> "
                        f"{after_count}"
                    ),
                )
            )

        # ==================================
        # 4. 不允许删除 Test Case
        #
        # 用 Counter，
        # 而不是 set。
        #
        # 防止同名测试重复出现时
        # 删除一个却没有被发现。
        # ==================================

        before_cases = self._extract_test_cases(failed_code)

        after_cases = self._extract_test_cases(repaired_code)

        removed_cases = before_cases - after_cases

        for (
            test_title,
            count,
        ) in sorted(removed_cases.items()):

            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_test_case_removed"),
                    relative_path=path,
                    message=(
                        "Repair 删除或重命名了"
                        "原有 Test Case: "
                        f"{test_title!r}; "
                        f"removed_count={count}"
                    ),
                )
            )

        # ==================================
        # 5. Assertion 数量不能减少
        #
        # v1 是 File-level。
        #
        # 后续如果需要更高精度，
        # 再做 AST / Test Case-level。
        # ==================================

        before_assertions = self._count_assertions(failed_code)

        after_assertions = self._count_assertions(repaired_code)

        if after_assertions < before_assertions:
            violations.append(
                VerificationRepairIntegrityViolation(
                    code=("repair_assertion_reduced"),
                    relative_path=path,
                    message=(
                        "Repair 减少了 assertion: "
                        f"{before_assertions}"
                        " -> "
                        f"{after_assertions}"
                    ),
                )
            )

        return violations

    @classmethod
    def _count_disabling_modifiers(
        cls,
        content: str,
    ) -> Counter[str]:

        return Counter(
            match.group("modifier")
            for match in cls._DISABLING_MODIFIER_PATTERN.finditer(content)
        )

    @classmethod
    def _extract_test_cases(
        cls,
        content: str,
    ) -> Counter[str]:
        """
        提取 Test Case Title。

        第一版只支持常见静态字符串 Title。

        例如：

            test('loads orders', ...)
            it("filters orders", ...)
            test(`shows error`, ...)

        动态 test.each 等高级形式
        后续进入 AST Backlog。
        """

        titles: list[str] = []

        for pattern in (
            cls._SINGLE_QUOTE_TEST_PATTERN,
            cls._DOUBLE_QUOTE_TEST_PATTERN,
            cls._BACKTICK_TEST_PATTERN,
        ):

            for match in pattern.finditer(content):

                title = match.group("title").strip()

                if title:
                    titles.append(title)

        return Counter(titles)

    @classmethod
    def _count_assertions(
        cls,
        content: str,
    ) -> int:

        expect_count = len(cls._EXPECT_PATTERN.findall(content))

        assert_count = len(cls._ASSERT_PATTERN.findall(content))

        return expect_count + assert_count

    @staticmethod
    def _strip_js_comments(
        content: str,
    ) -> str:
        """
        一个轻量 deterministic
        JS/TS comment stripper。

        目的不是解析 JavaScript AST。

        只是避免：

            // test.skip(...)
            /* expect(...) */

        这样的注释干扰 Integrity Guard。

        字符串和 template literal
        会原样保留。
        """

        result: list[str] = []

        index = 0

        length = len(content)

        quote: str | None = None

        escaped = False

        while index < length:

            char = content[index]

            next_char = content[index + 1] if (index + 1 < length) else ""

            # ==================================
            # 当前处于字符串内
            # ==================================

            if quote is not None:

                result.append(char)

                if escaped:
                    escaped = False

                elif char == "\\":
                    escaped = True

                elif char == quote:
                    quote = None

                index += 1
                continue

            # ==================================
            # String Start
            # ==================================

            if char in {
                "'",
                '"',
                "`",
            }:

                quote = char

                result.append(char)

                index += 1
                continue

            # ==================================
            # Line Comment
            # ==================================

            if char == "/" and next_char == "/":

                index += 2

                while index < length and content[index] not in {
                    "\n",
                    "\r",
                }:
                    index += 1

                # 保留换行，
                # 避免两行代码被拼在一起。
                continue

            # ==================================
            # Block Comment
            # ==================================

            if char == "/" and next_char == "*":

                index += 2

                while index < length - 1 and not (
                    content[index] == "*" and content[index + 1] == "/"
                ):

                    # 保留 block comment
                    # 中的换行。
                    if content[index] in {
                        "\n",
                        "\r",
                    }:
                        result.append(content[index])

                    index += 1

                if index < length - 1:
                    index += 2

                continue

            result.append(char)

            index += 1

        return "".join(result)
