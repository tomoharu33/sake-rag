from abc import ABC, abstractmethod
import json
import openai


# データをベクトル化するモジュールのインターフェース
class Embedder(ABC):

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, texts: list[str], filename: str) -> bool:
        raise NotImplementedError


# Embedderインターフェースの実装
class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def embed(self, texts: list[str]) -> list[list[float]]:
        # openai 1.10.0 で動作確認
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        # レスポンスからベクトルを抽出
        return [data.embedding for data in response.data]

    def save(self, texts: list[str], filename: str) -> bool:
        vectors = self.embed(texts)
        data_to_save = [
            {"id": idx, "text": text, "vector": vector}
            for idx, (text, vector) in enumerate(zip(texts, vectors))
        ]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=4)
        print(f"{filename} に保存されました。")
        return True


if __name__ == "__main__":
    import os

    texts = [
        "山廃・生酛は、醸造用乳酸を添加せずに乳酸菌を増殖させる伝統的な醸造方法である。1910年に速醸酛が開発されたことで、手間のかかる生酛や山廃は少数派になり、現状のシェアは速醸90%、山廃9%、生酛1%である。しかし、豊かで複雑みのある香味を求めて、生酛系酒母に取り組む蔵が目立ってきている。和食だけでなく、中華や洋食との相性も良いため、幅広いペアリングの対象として好まれる可能性がある。",
        "セルレニン耐性酵母は、リンゴ様の香り・カプロン酸エチルを多く生成する酵母で、「香り酵母」とも呼ばれる。1990年代中期に全国に広まった。代表的な酵母として、アルプス酵母やきょうかい1801号が挙げられる。香り酵母は全国各地で開発されており、地域色を打ち出したものも増えている。フルーティーな香りは日本酒初心者にも好まれ、新たな消費者が日本酒を飲むきっかけとなる可能性がある。",
        "村米制度は、酒造家と農家が直接契約して酒米を栽培する制度であり、「山田錦」の故郷である兵庫県では明治20年代から行われていた。農家は酒造家が好む酒米を生産するために品質向上を図る。テロワールによる集落ごとの格付けも行われ、集落内外での競争が活発化した。現在は「特A-a地区」と「特A-b地区」に分けられ、「特A-a地区」は吉川町、口吉川町、東条、社の91集落で構成されている。",
        "美山錦は、1978年に長野県農事試験場で「たかね錦」の種籾にγ線を照射して生み出された突然変異種の酒米である。醸造用玄米の中では「山田錦」「五百万石」に次ぎ生産量第３位を誇る。大粒で心白発現率が良いため、高精白が可能である。また耐冷性があるため、長野のほか東北地方が主な産地となっている。「亀ノ尾」など歴史ある品種を先祖にもち、「出羽燦々」「越の雫」「秋の精」など他県が開発した品種の親株でもある。",
        "奈良県は清酒発祥の地とされている。日本最古の神社・大神神社は酒造りの神で、杉玉の発祥の地でもある。奈良時代には造酒司が設けられ、酒造りの中心地となった。室町時代には酒母製法の一つである「菩提酛」が菩提山正暦寺で生み出された。菩提酛は「そやし水」と呼ばれる乳酸酸性水を使用して酒母を作る製法で、近年奈良県内の蔵元が再現している。酒米は自県産より他県からの移入が多いが、「露葉風」の生産量は日本一である。",
    ]

    # OpenAI APIキーを事前に環境変数にセットしてください。
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("APIキーがセットされていません。")

    embedder = OpenAIEmbedder(api_key)
    embedder.save(texts, "sample_data.json")