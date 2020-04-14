package ai.libs.asa;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ai.libs.jaicore.basic.ValueUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreCollection;
import ai.libs.jaicore.basic.kvstore.KVStoreStatisticsUtil;
import ai.libs.jaicore.basic.kvstore.KVStoreUtil;
import ai.libs.jaicore.db.sql.SQLAdapter;

public class SurvivalASTableGenerator {

	public static void main(final String[] args) throws SQLException {
		SQLAdapter adapter = new SQLAdapter("localhost", "", "", "conference_icml2020_survival");

		KVStoreCollection col = KVStoreUtil.readFromMySQLTable(adapter, "survival_vs_baselines_v01", new HashMap<>());

		KVStoreStatisticsUtil.rank(col, "scenario_name", "approach", "AVG_RESULT");
		KVStoreStatisticsUtil.best(col, "scenario_name", "approach", "AVG_RESULT");
		Map<String, DescriptiveStatistics> avgRankStats = KVStoreStatisticsUtil.averageRank(col, "approach", "rank");

		col.stream().forEach(x -> x.put("entry", (x.getAsBoolean("best") ? "\\textbf{" : "") + ValueUtil.round(x.getAsDouble("AVG_RESULT"), 1) + " (" + x.getAsString("rank") + ")" + (x.getAsBoolean("best") ? "}" : "")));

		String latexTable = KVStoreUtil.kvStoreCollectionToLaTeXTable(col, "scenario_name", "approach", "entry");
		latexTable = latexTable.replaceAll("\\\\_algorithm\\\\_survival\\\\_forest", " ASF");
		System.out.println(latexTable);

		avgRankStats.entrySet().stream().map(x -> x.getKey() + " " + x.getValue().getMean()).forEach(System.out::println);
	}

}
