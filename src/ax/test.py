import dataclasses


@dataclasses.dataclass
class Foo:
    def get_list(self):
        return [1, 2, 3]


def main():
    f = Foo()

    l1 = f.get_list()
    print(l1)

    l1.append(5)
    print(l1)

    print(f.get_list())

if __name__ == "__main__":
    main()
