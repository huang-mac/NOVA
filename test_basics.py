"""
Python 列表、元组、字典 基础练习题（10 道）
============================================
先自己做，再看答案。运行方式：python test_basics.py
"""


def sep(title=""):
    """打印分隔线，可附带标题。"""
    print()
    print("─" * 50)
    if title:
        print(f"  {title}")
        print("─" * 50)


# ================================================================
# 题目 1：列表基础
# ================================================================
sep("1. 列表基础")
print("""
【题目】
nums = [3, 1, 4, 1, 5, 9, 2, 6]
① 在末尾追加元素 7
② 在下标 2 位置插入 100
③ 删除第一个 1
④ 列表升序排序
⑤ 统计元素 1 出现次数
""")

input(">>> 按回车查看答案...")

nums = [3, 1, 4, 1, 5, 9, 2, 6]

nums.append(7)
print("① append(7) →", nums)
# 解析：append 把元素加到列表末尾，原地修改

nums.insert(2, 100)
print("② insert(2, 100) →", nums)
# 解析：insert(下标, 值) 在指定位置插入，原有元素后移

nums.remove(1)
print("③ remove(1) →", nums)
# 解析：remove(值) 删除第一个匹配项，找不到会报 ValueError

nums.sort()
print("④ sort() →", nums)
# 解析：sort() 原地升序排序，reverse=True 可降序

nums = [3, 1, 4, 1, 5, 9, 2, 6, 7]
count = nums.count(1)
print(f"⑤ count(1) → {count}")
# 解析：count(值) 统计元素在列表中出现的次数


# ================================================================
# 题目 2：列表遍历
# ================================================================
sep("2. 列表遍历")
print("""
【题目】
fruits = ["苹果", "香蕉", "橙子"]
用两种方式遍历：
  方式 1：直接 for 循环元素
  方式 2：带下标遍历（用 enumerate）
""")

input(">>> 按回车查看答案...")

fruits = ["苹果", "香蕉", "橙子"]

print("方式 1（直接遍历元素）：")
for f in fruits:
    print(f"  {f}")

print("方式 2（enumerate 带下标）：")
for i, f in enumerate(fruits):
    print(f"  [{i}] {f}")
# 解析：enumerate 返回 (下标, 元素) 元组，从 0 开始


# ================================================================
# 题目 3：元组操作
# ================================================================
sep("3. 元组操作")
print("""
【题目】
t = (2, 4, 6, 8, 10)
① 获取第 3 个元素
② 判断 5 是否在元组中
③ 统计 4 出现次数
④ 把元组转为列表
""")

input(">>> 按回车查看答案...")

t = (2, 4, 6, 8, 10)

print(f"① t[2] → {t[2]}")
# 解析：下标从 0 开始，第 3 个是下标 2

print(f"② 5 in t → {5 in t}")
# 解析：in 运算符判断元素是否存在

print(f"③ t.count(4) → {t.count(4)}")
# 解析：元组和列表一样，count(值) 统计出现次数

lst = list(t)
print(f"④ list(t) → {lst} (类型: {type(lst).__name__})")
# 解析：list() 可将任何可迭代对象转为列表


# ================================================================
# 题目 4：字典新增修改
# ================================================================
sep("4. 字典新增修改")
print("""
【题目】
user = {"name": "李四", "age": 22}
① 新增键值对 gender: "男"
② 修改年龄为 25
③ 用 setdefault 加 city: "北京"，已存在则不修改
""")

input(">>> 按回车查看答案...")

user = {"name": "李四", "age": 22}
print("初始 user =", user)

user["gender"] = "男"
print(f"① user['gender']='男' → {user}")
# 解析：dict[新键] = 值，直接赋值即新增

user["age"] = 25
print(f"② user['age']=25 → {user}")
# 解析：dict[已有键] = 值，会覆盖原值

result = user.setdefault("city", "北京")
result2 = user.setdefault("name", "张三")
print(f"③ setdefault('city','北京') 返回={result}, user={user}")
print(f"   setdefault('name','张三') 返回={result2}（已存在，不覆盖）")
# 解析：setdefault(key, default) — key 存在返回原值，不存在则设置


# ================================================================
# 题目 5：字典取值
# ================================================================
sep("5. 字典取值")
print("""
【题目】
沿用上面 user 字典
① 用 [] 取 name
② 用 get 取 age，取不到给默认值 18
③ 用 get 取不存在的键 score，默认返回 0
""")

input(">>> 按回车查看答案...")

user = {"name": "李四", "age": 25, "gender": "男", "city": "北京"}
print("user =", user)

print(f"① user['name'] → {user['name']}")
# 解析：[] 取值，键不存在会报 KeyError

print(f"② user.get('age', 18) → {user.get('age', 18)}")
# 解析：get(key, default)，key 存在返回原值，安全

print(f"③ user.get('score', 0) → {user.get('score', 0)}")
# 解析：key 不存在则返回默认值，不报错


# ================================================================
# 题目 6：字典删除
# ================================================================
sep("6. 字典删除")
print("""
【题目】
stu = {"id": 101, "name": "王五", "score": 88, "class": "一班"}
① 删除 class 键
② 随机删除最后一组键值对（popitem）
③ 清空整个字典（clear）
""")

input(">>> 按回车查看答案...")

stu = {"id": 101, "name": "王五", "score": 88, "class": "一班"}
print("初始 stu =", stu)

del stu["class"]
print(f"① del stu['class'] → {stu}")
# 解析：del dict[key] 删除指定键值对，键不存在报 KeyError

key, value = stu.popitem()
print(f"② popitem() → 删了 ({key}: {value}), 剩余 {stu}")
# 解析：popitem() 删除并返回"最后插入"的键值对（Python 3.7+ 有序）

stu.clear()
print(f"③ clear() → {stu}")
# 解析：clear() 清空所有键值对


# ================================================================
# 题目 7：字典遍历
# ================================================================
sep("7. 字典遍历")
print("""
【题目】
book = {"书名": "Python入门", "价格": 59, "作者": "张三"}
分别遍历：所有 key、所有 value、所有键值对
""")

input(">>> 按回车查看答案...")

book = {"书名": "Python入门", "价格": 59, "作者": "张三"}
print("book =", book)

print("所有 key：", end=" ")
for k in book.keys():
    print(k, end=" ")
print()

print("所有 value：", end=" ")
for v in book.values():
    print(v, end=" ")
print()

print("所有键值对：")
for k, v in book.items():
    print(f"  {k} → {v}")
# 解析：items() 返回 (key, value) 元组，最常用


# ================================================================
# 题目 8：列表去重
# ================================================================
sep("8. 列表去重")
print("""
【题目】
li = [2, 2, 3, 3, 4, 4, 5]
写代码去重，生成新列表。
""")

input(">>> 按回车查看答案...")

li = [2, 2, 3, 3, 4, 4, 5]
print("原始 li =", li)

new_li = list(set(li))
print(f"方法1 list(set(li)) → {new_li}（顺序可能打乱）")

new_li2 = []
for x in li:
    if x not in new_li2:
        new_li2.append(x)
print(f"方法2 遍历去重 → {new_li2}（保持原顺序）")

new_li3 = list(dict.fromkeys(li))
print(f"方法3 dict.fromkeys → {new_li3}（保持原顺序）")
# 解析：set 最快但不保证顺序，方法 2/3 保持原顺序


# ================================================================
# 题目 9：列表转字典
# ================================================================
sep("9. 列表转字典")
print("""
【题目】
keys = ["name", "age", "city"]
vals = ["赵六", 28, "广州"]
把两个列表合并成一个字典。
""")

input(">>> 按回车查看答案...")

keys = ["name", "age", "city"]
vals = ["赵六", 28, "广州"]
print("keys =", keys)
print("vals =", vals)

d = dict(zip(keys, vals))
print(f"方法1 dict(zip(keys, vals)) → {d}")

d2 = {keys[i]: vals[i] for i in range(len(keys))}
print(f"方法2 推导式 → {d2}")

d3 = {}
for k, v in zip(keys, vals):
    d3[k] = v
print(f"方法3 循环添加 → {d3}")
# 解析：zip(keys, vals) 返回 (key, value) 元组迭代器


# ================================================================
# 题目 10：综合小题
# ================================================================
sep("10. 综合小题")
print("""
【题目】
data = [
    {"name": "小明", "score": 75},
    {"name": "小红", "score": 92},
    {"name": "小刚", "score": 68},
]
遍历列表，打印分数大于 80 的人名。
""")

input(">>> 按回车查看答案...")

data = [
    {"name": "小明", "score": 75},
    {"name": "小红", "score": 92},
    {"name": "小刚", "score": 68},
]
print("data =", data)

print("结果：")
for item in data:
    if item["score"] > 80:
        print(f"  {item['name']}（分数：{item['score']}）")

high_scores = [item["name"] for item in data if item["score"] > 80]
print(f"列表推导式写法 → {high_scores}")

sep("全部完成")
print("总结 API 清单：")
print("  列表：append / insert / remove / sort / count / enumerate / [] 下标")
print("  元组：[] 下标 / in / count / list() 转换")
print("  字典：[] 增改 / setdefault / get / del / popitem / clear")
print("""  字典遍历：keys() / values() / items()
  通用：list() / set() / dict() / zip() / in""")
