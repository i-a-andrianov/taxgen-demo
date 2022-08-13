import { useObservable } from "rxjs-hooks";
import TaxonomyModel from "./TaxonomyModel";
import TaxonomyView from "./TaxonomyView";

const model = TaxonomyModel();

export default function TaxonomyController() {
    const taxonomy = useObservable(() => model.taxonomy$);
    const misc = useObservable(() => model.misc$);

    if (!taxonomy || !misc) {
        return (<>Loading...</>);
    }
    return (
        <TaxonomyView taxonomy={taxonomy}
            misc={misc}
            navigateToRoot={model.navigateToRoot}
            navigateToWord={model.navigateToWord}
            navigateToSearch={model.navigateToSearch}
            generateWords={model.generateWords}
            generateRelations={model.generateRelations}
            regenerateGraph={model.regenerateGraph}/>
    );
}
