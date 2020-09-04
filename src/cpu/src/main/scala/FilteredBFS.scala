package benchmark

import java.lang.System.currentTimeMillis

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType

import scala.collection.mutable


class FilteredBFS(session: PgxSession, graph: PgxGraph, source: Long) extends Algorithm  {

  val analyst = session.createAnalyst()

  val srcFBFS = classOf[FilteredBFS].getResourceAsStream("/fbfs.gm")
  val fbfs = session.compileProgram(srcFBFS)  


  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Compute FilteredBFS using Green-Marl on an input graph.
    */
  def compute(): Long = {

    if (!graph.getVertexProperties.contains("dist")) {
      graph.createVertexProperty(PropertyType.INTEGER, "dist")
    } else{
      graph.getVertexProperty("dist").fill(0)
    }

    if (!graph.getVertexProperties.contains("prop")) {
      graph.createVertexProperty(PropertyType.INTEGER, "prop")
    } else{
      graph.getVertexProperty("prop").fill(0)
    }

    var start = currentTimeMillis
    fbfs.run(graph, graph.getVertex(source), graph.getVertexProperty("dist"), graph.getVertexProperty("prop"))

    return currentTimeMillis - start
  }
}
