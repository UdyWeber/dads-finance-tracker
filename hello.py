from decimal import Decimal
import re
from typing import Any
import ollama
from pydantic import BaseModel, Field, model_validator

PATH = "samples/cleanest.jpg"


class Spense(BaseModel):
    name: str
    amount: Decimal


class Page(BaseModel):
    month: str
    total: Decimal
    expenses: list[Spense] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_sum(self: "Page") -> "Page":
        expenses_sum = sum([s.amount for s in self.expenses])

        if self.total != expenses_sum:
            print(f"Didnt matched total: {self.total} with sum: {expenses_sum}")
            raise ValueError("Total amount of page does not match with spenses sum")

        return self


def get_model_analysis(previsou_stack_trace: list[str] = []) -> Page:
    completed = []

    if previsou_stack_trace:
        previous_trace = "\n".join(previsou_stack_trace)

        error_gens = [
            {
                "role": "system",
                "content": """
                    This is not the first iteration of your generation.
                    You've generated something that does not conform
                    with the rules
                """,
            },
            {
                "role": "system",
                "content": """
                    - Look for fiels that might not have been read
                    - Reread the entire file and rescan the fields
                    - Change all formating from dots to commas
                    - Might not read correctly the month or total value
                    - Do not mix previous generations with current
                """,
                "images": [PATH],
            },
            {
                "role": "system",
                "content": f"Previous stack traces: {previous_trace}",
            },
        ]
        completed.extend(error_gens)

    messages = [
        {
            "role": "system",
            "content": """
                    Você é uma ferramenta de reconhecimento de texto
                    cursivo em imagens de cadernos escritos a mão.
                """,
        },
        {
            "role": "system",
            "content": """
                Informações contindas nas páginas:
                    - Preço total descrito em reais do gasto total do mês.
                    - Nome do mês em questão, logo abaixo do gasto total do mês
                    - Lista de gastos feitos naquele mes.
                        Exemplo `{NOME_GASTO} R$ = {PRECO_GASTO}`
                """,
        },
        {
            "role": "system",
            "content": """
                Despesas regulares:
                    Luz
                    Agua
                    Fink
                    Telefone
                    Fachineira
                    Atacadao
                """,
        },
        {"role": "system", "content": "Only output the result, no explanations"},
        {
            "role": "user",
            "content": """
                    Transcreva o conteudo da imagem sem alucinar,
                    mantenha coesao, tudo linha por linha.
                """,
            "images": [PATH],
        },
    ]

    completed.extend(messages)
    response = ollama.chat(model="llama3.2-vision", messages=completed)

    model = None

    try:
        print("Processing", response.message.content.__str__())
        clean = clean_data(response.message.content.__str__())
        model = Page(**clean)
    except Exception as e:
        print(e.__str__())
        previsou_stack_trace.append(e.__str__())
        get_model_analysis(previsou_stack_trace)

    return model


def clean_data(text: str) -> dict[str, Any]:
    lines = list(filter(lambda x: x != "", text.split("\n")))
    result: dict[str, Any] = {
        "month": lines[0],
        "total": lines[1].split("=")[1].replace(".", "").replace(",", "").strip(),
    }
    spenses: list[Any] = []

    regex = r"^([A-Za-zÀ-ÖØ-öø-ÿ\s]+)\sR\$\s=\s([\d,.\s]+)$"
    for line in lines[2:]:
        matches = re.match(regex, line.strip())

        if not matches:
            print(f"Pattern didnt match: {line}")
            continue

        name = matches.group(1)
        entries = matches.group(2).strip().split(",")
        total = sum([Decimal(e) for e in entries if e != ""])

        spenses.append({"name": name, "amount": total})

    result["expenses"] = spenses

    return result


def main():
    response = get_model_analysis()
    assert response is not None
    print(response.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
