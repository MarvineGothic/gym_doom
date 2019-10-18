class CFG_Parser:
    @staticmethod
    def parse(fileName):
        result = {}
        inner_array = []
        dict_key = ""

        file = open(fileName, "r")
        for x in file:
            x = x.strip()
            if x == '' or x.startswith(" ") or x.startswith("#"):
                pass
            else:
                if dict_key == "":
                    y = x.split("=")
                    key = y[0].strip()
                    value = y[1].strip()

                    if "{" in value and "}" in value:
                        value = value.replace("{", "").strip()
                        value = value.replace("}", "").strip()

                        if " " in value:
                            values = value.split(" ")
                            for i in range(len(values)):
                                if values[i] != "":
                                    inner_array.append(values[i].strip())
                            result[key] = inner_array
                            inner_array = []
                        else:
                            result[key] = [value]

                    elif value == "" or "{" in value:
                        dict_key = key
                    else:
                        result[key] = value
                else:
                    if x == "}":
                        result[dict_key] = inner_array
                        inner_array = []
                        dict_key = ""
                        pass
                    elif x != "{" and x != "":
                        inner_array.append(x)

        file.close()
        return result
